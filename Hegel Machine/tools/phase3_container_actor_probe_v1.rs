//! Independent Rust live probe for the Phase-3 container actor profile.
//!
//! This source is compiled once by the pinned Rust image with networking
//! disabled.  The resulting binary is then mounted read-only into purpose 3.
//! It uses no crates and emits exactly one compact JSON line.

use std::collections::BTreeMap;
use std::env;
use std::ffi::{c_char, c_int, c_long, c_uint, c_ulong, c_void};
use std::fs::{self, OpenOptions};
use std::io::Error;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::time::Duration;

const SCHEMA: &str = "hegel-container-actor-live-probe/1";
const IMPLEMENTATION: &str = "rust-ffi-v1";
const PROFILE_ENV: &str = "HEGEL_ACTOR_PROFILE_ID";
const PURPOSE_ENV: &str = "HEGEL_PURPOSE_ID";

const AF_INET: c_int = 2;
const AF_INET6: c_int = 10;
const SOCK_STREAM: c_int = 1;
const SYS_PERF_EVENT_OPEN: c_long = 298;
const SYS_BPF: c_long = 321;

unsafe extern "C" {
    fn socket(domain: c_int, socket_type: c_int, protocol: c_int) -> c_int;
    fn close(fd: c_int) -> c_int;
    fn mount(
        source: *const c_char,
        target: *const c_char,
        filesystem_type: *const c_char,
        flags: c_ulong,
        data: *const c_void,
    ) -> c_int;
    fn ptrace(request: c_uint, ...) -> c_long;
    fn syscall(number: c_long, ...) -> c_long;
    fn getuid() -> c_uint;
    fn getgid() -> c_uint;
    fn getpid() -> c_int;
}

#[derive(Clone)]
struct ProbeRow {
    probe_id: &'static str,
    return_value: i64,
    errno: i32,
}

fn json_string(value: &str) -> String {
    let mut result = String::from("\"");
    for ch in value.chars() {
        match ch {
            '\"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if (c as u32) < 0x20 => result.push_str(&format!("\\u{:04x}", c as u32)),
            c => result.push(c),
        }
    }
    result.push('\"');
    result
}

fn errno_result(value: c_long) -> (i64, i32) {
    let observed_errno = if value == -1 {
        Error::last_os_error().raw_os_error().unwrap_or(0)
    } else {
        0
    };
    (value as i64, observed_errno)
}

fn syscall_rows() -> Vec<ProbeRow> {
    let mut rows = Vec::new();
    unsafe {
        let fd4 = socket(AF_INET, SOCK_STREAM, 0);
        let (rv4, err4) = errno_result(fd4 as c_long);
        if fd4 >= 0 {
            close(fd4);
        }
        rows.push(ProbeRow {
            probe_id: "socket(AF_INET, SOCK_STREAM)",
            return_value: rv4,
            errno: err4,
        });

        let fd6 = socket(AF_INET6, SOCK_STREAM, 0);
        let (rv6, err6) = errno_result(fd6 as c_long);
        if fd6 >= 0 {
            close(fd6);
        }
        rows.push(ProbeRow {
            probe_id: "socket(AF_INET6, SOCK_STREAM)",
            return_value: rv6,
            errno: err6,
        });

        let target = b"/tmp/hegel-mount-probe\0";
        let mount_value = mount(
            std::ptr::null(),
            target.as_ptr() as *const c_char,
            std::ptr::null(),
            0,
            std::ptr::null(),
        );
        let (rv_mount, err_mount) = errno_result(mount_value as c_long);
        rows.push(ProbeRow {
            probe_id: "mount",
            return_value: rv_mount,
            errno: err_mount,
        });

        let ptrace_value = ptrace(
            0_u32,
            std::ptr::null_mut::<c_void>(),
            std::ptr::null_mut::<c_void>(),
            std::ptr::null_mut::<c_void>(),
        );
        let (rv_ptrace, err_ptrace) = errno_result(ptrace_value);
        rows.push(ProbeRow {
            probe_id: "ptrace(PTRACE_TRACEME)",
            return_value: rv_ptrace,
            errno: err_ptrace,
        });

        let bpf_value = syscall(
            SYS_BPF,
            0 as c_long,
            std::ptr::null_mut::<c_void>(),
            0 as c_long,
        );
        let (rv_bpf, err_bpf) = errno_result(bpf_value);
        rows.push(ProbeRow {
            probe_id: "bpf(BPF_MAP_CREATE)",
            return_value: rv_bpf,
            errno: err_bpf,
        });

        let mut perf_attr = [0_u8; 128];
        let perf_value = syscall(
            SYS_PERF_EVENT_OPEN,
            perf_attr.as_mut_ptr(),
            0 as c_long,
            -1 as c_long,
            -1 as c_long,
            0 as c_long,
        );
        let (rv_perf, err_perf) = errno_result(perf_value);
        rows.push(ProbeRow {
            probe_id: "perf_event_open",
            return_value: rv_perf,
            errno: err_perf,
        });
    }
    rows
}

fn proc_status() -> BTreeMap<String, String> {
    let required = [
        "CapInh",
        "CapPrm",
        "CapEff",
        "CapBnd",
        "CapAmb",
        "NoNewPrivs",
        "Seccomp",
    ];
    let text = fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut result = BTreeMap::new();
    for line in text.lines() {
        if let Some((key, value)) = line.split_once(':') {
            if required.contains(&key) {
                result.insert(key.to_string(), value.trim().to_string());
            }
        }
    }
    result
}

fn namespace_rows() -> BTreeMap<String, String> {
    let mut result = BTreeMap::new();
    for kind in ["pid", "mnt", "net", "ipc", "uts"] {
        let value = fs::read_link(format!("/proc/self/ns/{kind}"))
            .map(|path| path.to_string_lossy().to_string())
            .unwrap_or_default();
        result.insert(kind.to_string(), value);
    }
    result
}

fn network_interfaces() -> Vec<String> {
    let mut result = fs::read_dir("/sys/class/net")
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .filter_map(|entry| entry.file_name().into_string().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    result.sort();
    result
}

fn open_fds() -> Vec<i32> {
    let names = fs::read_dir("/proc/self/fd")
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .filter_map(|entry| entry.file_name().into_string().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut result = Vec::new();
    for name in names {
        let path = format!("/proc/self/fd/{name}");
        if fs::read_link(path).is_ok() {
            if let Ok(fd) = name.parse::<i32>() {
                result.push(fd);
            }
        }
    }
    result.sort();
    result
}

fn write_denial(path: &str) -> (bool, i32) {
    match OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open(path)
    {
        Ok(file) => {
            drop(file);
            let _ = fs::remove_file(path);
            (false, 0)
        }
        Err(error) => (true, error.raw_os_error().unwrap_or(0)),
    }
}

fn existing_paths(paths: &[&str]) -> Vec<String> {
    paths
        .iter()
        .filter(|path| Path::new(path).exists())
        .map(|path| path.to_string())
        .collect()
}

fn string_array(values: &[String]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| json_string(value))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn main() {
    let profile_id = env::var(PROFILE_ENV).unwrap_or_default();
    let purpose_id = env::var(PURPOSE_ENV)
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(-1);
    let input_probe_path = env::var("HEGEL_PROBE_INPUT_WRITE_PATH")
        .unwrap_or_else(|_| "/actor_input/profile.json".to_string());
    let linger_seconds = env::var("HEGEL_PROBE_LINGER_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    let mut raw_path_visible = false;
    if let Ok(host_repository_path) = env::var("HEGEL_HOST_REPOSITORY_PATH") {
        raw_path_visible = !host_repository_path.is_empty()
            && Path::new(&host_repository_path).exists();
    }
    // The raw clone path is probe-only.  Remove it from the actual Rust
    // process environment before the remaining probes, serialization, or
    // lingering can create a descendant that inherits it.
    env::remove_var("HEGEL_HOST_REPOSITORY_PATH");
    let status = proc_status();
    let namespaces = namespace_rows();
    let interfaces = network_interfaces();
    let rows = syscall_rows();
    let (root_denied, root_errno) = write_denial("/hegel-container-root-write-probe");
    let (input_denied, input_errno) = write_denial(&input_probe_path);
    let forbidden = existing_paths(&[
        "/var/run/docker.sock",
        "/run/docker.sock",
        "/workspace",
        "/repo",
        "/mnt/c",
    ]);
    let mut forbidden = forbidden;
    if raw_path_visible {
        // Bind failure to a stable label without disclosing the host path.
        forbidden.push("HEGEL_HOST_REPOSITORY_PATH".to_string());
    }
    let cross = existing_paths(&[
        "/purpose-1",
        "/purpose-2",
        "/purpose-3",
        "/purpose-4",
        "/actor-1",
        "/actor-2",
        "/actor-3",
        "/actor-4",
    ]);
    let environment = env::vars().collect::<BTreeMap<_, _>>();
    let fds = open_fds();

    let status_json = status
        .iter()
        .map(|(key, value)| {
            if key == "NoNewPrivs" || key == "Seccomp" {
                format!("{}:{}", json_string(key), value)
            } else {
                format!("{}:{}", json_string(key), json_string(value))
            }
        })
        .collect::<Vec<_>>()
        .join(",");
    let namespace_json = namespaces
        .iter()
        .map(|(key, value)| format!("{}:{}", json_string(key), json_string(value)))
        .collect::<Vec<_>>()
        .join(",");
    let syscall_json = rows
        .iter()
        .map(|row| {
            format!(
                "{{\"errno\":{},\"probe_id\":{},\"return_value\":{}}}",
                row.errno,
                json_string(row.probe_id),
                row.return_value
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let environment_json = environment
        .iter()
        .map(|(key, value)| format!("{}:{}", json_string(key), json_string(value)))
        .collect::<Vec<_>>()
        .join(",");
    let fd_json = fds
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",");

    // uid/gid are read from libc; proc metadata is touched above only to keep
    // the Rust implementation independent of the Python implementation.
    let (uid, gid, pid) = unsafe { (getuid(), getgid(), getpid()) };
    let _proc_uid = fs::metadata("/proc/self").map(|meta| meta.uid()).ok();
    println!(
        "{{\"environment\":{{{environment_json}}},\"filesystem_probes\":{{\"cross_purpose_paths_present\":{},\"forbidden_paths_present\":{},\"input_write\":{{\"denied\":{input_denied},\"errno\":{input_errno}}},\"root_write\":{{\"denied\":{root_denied},\"errno\":{root_errno}}}}},\"identity\":{{\"gid\":{gid},\"pid\":{pid},\"uid\":{uid}}},\"implementation\":{},\"namespaces\":{{{namespace_json}}},\"network_interfaces\":{},\"open_fds\":[{fd_json}],\"proc_status\":{{{status_json}}},\"profile_id\":{},\"purpose_id\":{purpose_id},\"schema\":{},\"syscall_probes\":[{syscall_json}]}}",
        string_array(&cross),
        string_array(&forbidden),
        json_string(IMPLEMENTATION),
        string_array(&interfaces),
        json_string(&profile_id),
        json_string(SCHEMA),
    );
    std::thread::sleep(Duration::from_secs(linger_seconds));
}
