use hegel_q1_archive_projection_oracle::{
    actor_error_json, bounded_node3_actor_emission, embedded_source_identity_sha256,
    runtime_identity_sha256, ActorEmission, OracleError, ACTOR_ACTION_ID,
    OUTPUT_RELATIVE_PATHS,
};
use std::collections::BTreeSet;
use std::ffi::CString;
use std::fs::{self, File};
use std::io::{self, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const MAXIMUM_SIDECAR_BYTES: usize = 64 * 1024 * 1024;

struct OutputDirectory {
    path: PathBuf,
    descriptor: OwnedFd,
    device: u64,
    inode: u64,
}

fn action_error() -> OracleError {
    OracleError::new(
        "Q1_PROJECTION_ACTION_NOT_ADMITTED",
        format!(
            "the only admitted invocation is --action {ACTOR_ACTION_ID} --output-dir ABSOLUTE_EMPTY_DIR"
        ),
    )
}

fn parse_invocation(arguments: &[String]) -> Result<PathBuf, OracleError> {
    if arguments.len() != 4
        || arguments[0] != "--action"
        || arguments[1] != ACTOR_ACTION_ID
        || arguments[2] != "--output-dir"
    {
        return Err(action_error());
    }
    let path = PathBuf::from(&arguments[3]);
    if !path.is_absolute() {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_OUTPUT_DIR",
            "output directory must be absolute",
        ));
    }
    Ok(path)
}

fn c_path(path: &Path) -> Result<CString, OracleError> {
    CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        OracleError::new(
            "FAIL_Q1_PROJECTION_OUTPUT_DIR",
            "output directory path contains NUL",
        )
    })
}

fn open_directory_path(path: &Path) -> Result<OwnedFd, OracleError> {
    let raw = unsafe {
        libc::open(
            c_path(path)?.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if raw < 0 {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_OUTPUT_DIR",
            format!("cannot securely open output directory: {}", io::Error::last_os_error()),
        ));
    }
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

fn open_child_directory(parent: RawFd, name: &str) -> Result<OwnedFd, OracleError> {
    let name = CString::new(name).expect("frozen child directory name");
    let raw = unsafe {
        libc::openat(
            parent,
            name.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if raw < 0 {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_SIDECAR_WRITE",
            format!("cannot securely open output child: {}", io::Error::last_os_error()),
        ));
    }
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

fn descriptor_path(descriptor: RawFd) -> PathBuf {
    PathBuf::from(format!("/proc/self/fd/{descriptor}"))
}

fn entry_names(descriptor: RawFd) -> Result<BTreeSet<String>, OracleError> {
    let mut names = BTreeSet::new();
    for entry in fs::read_dir(descriptor_path(descriptor)).map_err(|error| {
        OracleError::new(
            "FAIL_Q1_PROJECTION_OUTPUT_DIR",
            format!("cannot list pinned output directory: {error}"),
        )
    })? {
        let entry = entry.map_err(|error| {
            OracleError::new(
                "FAIL_Q1_PROJECTION_OUTPUT_DIR",
                format!("cannot read pinned output entry: {error}"),
            )
        })?;
        let name = entry.file_name().into_string().map_err(|_| {
            OracleError::new(
                "FAIL_Q1_PROJECTION_OUTPUT_DIR",
                "output directory contains a non-UTF-8 entry",
            )
        })?;
        if !names.insert(name) {
            return Err(OracleError::new(
                "FAIL_Q1_PROJECTION_OUTPUT_DIR",
                "output directory contains a duplicate entry",
            ));
        }
    }
    Ok(names)
}

fn open_output_directory(path: PathBuf) -> Result<OutputDirectory, OracleError> {
    let metadata = fs::symlink_metadata(&path).map_err(|error| {
        OracleError::new(
            "FAIL_Q1_PROJECTION_OUTPUT_DIR",
            format!("cannot stat output directory: {error}"),
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_OUTPUT_DIR",
            "output directory must be one existing nonsymlink directory",
        ));
    }
    let canonical = fs::canonicalize(&path).map_err(|error| {
        OracleError::new(
            "FAIL_Q1_PROJECTION_OUTPUT_DIR",
            format!("cannot canonicalize output directory: {error}"),
        )
    })?;
    if canonical != path {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_OUTPUT_DIR",
            "output directory must be an exact canonical path",
        ));
    }
    let descriptor = open_directory_path(&path)?;
    let pinned = fs::metadata(descriptor_path(descriptor.as_raw_fd())).map_err(|error| {
        OracleError::new(
            "FAIL_Q1_PROJECTION_OUTPUT_DIR",
            format!("cannot stat pinned output directory: {error}"),
        )
    })?;
    if metadata.dev() != pinned.dev() || metadata.ino() != pinned.ino() {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_OUTPUT_DIR",
            "output directory identity changed while opening",
        ));
    }
    if !entry_names(descriptor.as_raw_fd())?.is_empty() {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_OUTPUT_DIR",
            "output directory must be empty",
        ));
    }
    Ok(OutputDirectory {
        path,
        descriptor,
        device: metadata.dev(),
        inode: metadata.ino(),
    })
}

fn mkdir_at(parent: RawFd, name: &str) -> Result<(), OracleError> {
    let name = CString::new(name).expect("frozen child directory name");
    let result = unsafe { libc::mkdirat(parent, name.as_ptr(), 0o700) };
    if result != 0 {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_SIDECAR_WRITE",
            format!("cannot create output child: {}", io::Error::last_os_error()),
        ));
    }
    Ok(())
}

fn fsync_descriptor(descriptor: RawFd) -> Result<(), OracleError> {
    if unsafe { libc::fsync(descriptor) } != 0 {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_SIDECAR_WRITE",
            format!("fsync failed: {}", io::Error::last_os_error()),
        ));
    }
    Ok(())
}

fn create_exclusive_file(
    parent: RawFd,
    name: &str,
    payload: &[u8],
) -> Result<(), OracleError> {
    let name = CString::new(name).expect("frozen output filename");
    let raw = unsafe {
        libc::openat(
            parent,
            name.as_ptr(),
            libc::O_WRONLY
                | libc::O_CREAT
                | libc::O_EXCL
                | libc::O_CLOEXEC
                | libc::O_NOFOLLOW,
            0o600,
        )
    };
    if raw < 0 {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_SIDECAR_WRITE",
            format!("exclusive sidecar creation failed: {}", io::Error::last_os_error()),
        ));
    }
    let mut file = unsafe { File::from_raw_fd(raw) };
    file.write_all(payload).map_err(|error| {
        OracleError::new(
            "FAIL_Q1_PROJECTION_SIDECAR_WRITE",
            format!("sidecar write failed: {error}"),
        )
    })?;
    if unsafe { libc::fchmod(file.as_raw_fd(), 0o444) } != 0 {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_SIDECAR_WRITE",
            format!("sidecar chmod failed: {}", io::Error::last_os_error()),
        ));
    }
    file.sync_all().map_err(|error| {
        OracleError::new(
            "FAIL_Q1_PROJECTION_SIDECAR_WRITE",
            format!("sidecar fsync failed: {error}"),
        )
    })?;
    Ok(())
}

fn verify_published_file(
    directory: RawFd,
    filename: &str,
    expected_length: usize,
) -> Result<(), OracleError> {
    let path = descriptor_path(directory).join(filename);
    let metadata = fs::symlink_metadata(&path).map_err(|error| {
        OracleError::new(
            "FAIL_Q1_PROJECTION_SIDECAR_SET",
            format!("cannot stat published sidecar: {error}"),
        )
    })?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.mode() & 0o777 != 0o444
        || metadata.len() != expected_length as u64
    {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_SIDECAR_SET",
            "published sidecar type, mode, or length differs",
        ));
    }
    Ok(())
}

fn publish_sidecars(
    output: &OutputDirectory,
    emission: &ActorEmission,
) -> Result<(), OracleError> {
    if emission
        .files
        .iter()
        .map(|file| file.relative_path)
        .ne(OUTPUT_RELATIVE_PATHS)
        || emission.files.iter().any(|file| file.payload.is_empty())
    {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_SIDECAR_SET",
            "sidecar path registry, order, or payload differs",
        ));
    }
    let total = emission.files.iter().try_fold(0_usize, |sum, file| {
        sum.checked_add(file.payload.len()).ok_or_else(|| {
            OracleError::new(
                "INCONCLUSIVE_Q1_PROJECTION_OUTPUT_LIMIT",
                "sidecar byte total overflow",
            )
        })
    })?;
    if total > MAXIMUM_SIDECAR_BYTES {
        return Err(OracleError::new(
            "INCONCLUSIVE_Q1_PROJECTION_OUTPUT_LIMIT",
            "sidecar byte total exceeds 64 MiB",
        ));
    }
    let root_fd = output.descriptor.as_raw_fd();
    let pinned = fs::metadata(descriptor_path(root_fd)).map_err(|error| {
        OracleError::new(
            "FAIL_Q1_PROJECTION_OUTPUT_DIR",
            format!("cannot re-stat pinned output directory: {error}"),
        )
    })?;
    if pinned.dev() != output.device
        || pinned.ino() != output.inode
        || !entry_names(root_fd)?.is_empty()
    {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_OUTPUT_DIR",
            "output directory identity or emptiness changed",
        ));
    }

    mkdir_at(root_fd, "preimages")?;
    mkdir_at(root_fd, "neutral")?;
    let preimages = open_child_directory(root_fd, "preimages")?;
    let neutral = open_child_directory(root_fd, "neutral")?;
    for file in &emission.files {
        let (directory, filename) = file.relative_path.split_once('/').ok_or_else(|| {
            OracleError::new(
                "FAIL_Q1_PROJECTION_SIDECAR_SET",
                "sidecar path is not exactly one child and filename",
            )
        })?;
        let descriptor = match directory {
            "preimages" => preimages.as_raw_fd(),
            "neutral" => neutral.as_raw_fd(),
            _ => {
                return Err(OracleError::new(
                    "FAIL_Q1_PROJECTION_SIDECAR_SET",
                    "sidecar directory is unregistered",
                ))
            }
        };
        create_exclusive_file(descriptor, filename, &file.payload)?;
    }
    fsync_descriptor(preimages.as_raw_fd())?;
    fsync_descriptor(neutral.as_raw_fd())?;
    fsync_descriptor(root_fd)?;

    let expected_root = BTreeSet::from(["neutral".to_owned(), "preimages".to_owned()]);
    let expected_preimages = BTreeSet::from([
        "000-full-v16-leaf-manifest-v1.cbor".to_owned(),
        "001-odd-node3-partition-evidence-v1.cbor".to_owned(),
        "002-sink-node3-partition-evidence-v1.cbor".to_owned(),
    ]);
    let expected_neutral = BTreeSet::from([
        "q05b-node3-golden-manifest-v1.cbor".to_owned(),
        "q05b-node3-sidecar-manifest-v1.cbor".to_owned(),
    ]);
    if entry_names(root_fd)? != expected_root
        || entry_names(preimages.as_raw_fd())? != expected_preimages
        || entry_names(neutral.as_raw_fd())? != expected_neutral
    {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_SIDECAR_SET",
            "published output tree differs from the exact five-file registry",
        ));
    }
    for file in &emission.files {
        let (directory, filename) = file.relative_path.split_once('/').unwrap();
        let descriptor = if directory == "preimages" {
            preimages.as_raw_fd()
        } else {
            neutral.as_raw_fd()
        };
        verify_published_file(descriptor, filename, file.payload.len())?;
    }
    fsync_descriptor(root_fd)?;
    Ok(())
}

fn run_actor() -> Result<Vec<u8>, OracleError> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let output_path = parse_invocation(&arguments)?;
    let output = open_output_directory(output_path)?;
    let source_identity = embedded_source_identity_sha256()?;
    let runtime_identity = runtime_identity_sha256()?;
    let emission = bounded_node3_actor_emission(source_identity, &runtime_identity)?;
    if runtime_identity_sha256()? != runtime_identity {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_IDENTITY_CHANGED",
            "runtime executable changed during actor execution",
        ));
    }
    publish_sidecars(&output, &emission)?;
    if fs::canonicalize(&output.path).map_err(|error| {
        OracleError::new(
            "FAIL_Q1_PROJECTION_OUTPUT_DIR",
            format!("cannot revalidate output path: {error}"),
        )
    })? != output.path
    {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_OUTPUT_DIR",
            "output path changed after publication",
        ));
    }
    Ok(emission.stdout)
}

fn write_stdout(payload: &[u8]) -> bool {
    let mut stdout = io::stdout().lock();
    stdout.write_all(payload).is_ok() && stdout.flush().is_ok()
}

fn main() -> ExitCode {
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(run_actor).unwrap_or_else(|_| {
        Err(OracleError::new(
            "FAIL_Q1_PROJECTION_UNHANDLED",
            "panic caught by the fail-closed actor boundary",
        ))
    });
    match result {
        Ok(payload) if write_stdout(&payload) => ExitCode::SUCCESS,
        Ok(_) => ExitCode::from(1),
        Err(error) => {
            let _ = write_stdout(&actor_error_json(&error));
            ExitCode::from(1)
        }
    }
}
