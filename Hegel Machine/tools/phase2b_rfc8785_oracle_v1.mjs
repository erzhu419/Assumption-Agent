#!/usr/bin/env node
/* Independent RFC 8785 oracle using only Node/V8 and built-in modules.
 *
 * This file intentionally does not import the Python production module.  V8's
 * JSON.stringify supplies the normative ECMAScript binary64 and string
 * spelling; this wrapper adds I-JSON validation and recursive UTF-16 sorting.
 */

import fs from 'fs';


class OracleError extends Error {}


function isNoncharacter(codepoint) {
  return (codepoint >= 0xfdd0 && codepoint <= 0xfdef) ||
         (codepoint & 0xffff) === 0xfffe ||
         (codepoint & 0xffff) === 0xffff;
}


function validateString(value) {
  for (let index = 0; index < value.length; index += 1) {
    const first = value.charCodeAt(index);
    let codepoint = first;
    if (first >= 0xd800 && first <= 0xdbff) {
      if (index + 1 >= value.length) {
        throw new OracleError('lone high surrogate');
      }
      const second = value.charCodeAt(index + 1);
      if (second < 0xdc00 || second > 0xdfff) {
        throw new OracleError('lone high surrogate');
      }
      codepoint = 0x10000 + ((first - 0xd800) << 10) + second - 0xdc00;
      index += 1;
    } else if (first >= 0xdc00 && first <= 0xdfff) {
      throw new OracleError('lone low surrogate');
    }
    if (isNoncharacter(codepoint)) {
      throw new OracleError('Unicode noncharacter');
    }
  }
}


function canonicalize(value, active = new Set()) {
  if (value === null) {
    return 'null';
  }
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false';
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new OracleError('non-finite number');
    }
    return JSON.stringify(value);
  }
  if (typeof value === 'string') {
    validateString(value);
    return JSON.stringify(value);
  }
  if (typeof value !== 'object') {
    throw new OracleError('value is outside JSON data model');
  }
  if (active.has(value)) {
    throw new OracleError('cyclic JSON value');
  }
  active.add(value);
  try {
    if (Array.isArray(value)) {
      return '[' + value.map((element) => canonicalize(element, active)).join(',') + ']';
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new OracleError('non-plain object');
    }
    const keys = Object.keys(value);
    keys.forEach(validateString);
    keys.sort();
    return '{' + keys.map((key) =>
      JSON.stringify(key) + ':' + canonicalize(value[key], active)
    ).join(',') + '}';
  } finally {
    active.delete(value);
  }
}


function numberFromBits(hex) {
  if (!/^[0-9a-f]{16}$/.test(hex)) {
    throw new OracleError('binary64 bits must be 16 lowercase hexadecimal digits');
  }
  return Buffer.from(hex, 'hex').readDoubleBE(0);
}


function requireEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new OracleError(`${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}


function checkVectors(path) {
  const vectors = JSON.parse(fs.readFileSync(path, 'utf8'));
  let serializationCases = 0;
  let numberCases = 0;
  let rejectionCases = 0;

  for (const [label, vector] of [
    ['official Appendix sample', vectors.official_appendix_sample],
    ['official property sorting', vectors.official_property_sorting],
    ...vectors.independent_literal_vectors.map((item) => [item.name, item]),
  ]) {
    requireEqual(canonicalize(JSON.parse(vector.input_json)), vector.canonical, label);
    serializationCases += 1;
  }

  for (const vector of vectors.official_appendix_b_numbers) {
    const value = numberFromBits(vector.ieee754_hex);
    if (Object.prototype.hasOwnProperty.call(vector, 'error')) {
      let rejected = false;
      try {
        canonicalize(value);
      } catch (error) {
        rejected = error instanceof OracleError;
      }
      if (!rejected) {
        throw new OracleError(`${vector.ieee754_hex}: expected rejection`);
      }
      rejectionCases += 1;
    } else {
      requireEqual(canonicalize(value), vector.canonical, vector.ieee754_hex);
      numberCases += 1;
    }
  }

  process.stdout.write(JSON.stringify({
    disposition: 'PASS',
    oracle: 'node-v8-json-stringify',
    node: process.version,
    v8: process.versions.v8,
    serialization_cases: serializationCases,
    number_cases: numberCases,
    rejection_cases: rejectionCases,
  }) + '\n');
}


function main(argv) {
  if (argv.length === 2 && argv[0] === '--check') {
    checkVectors(argv[1]);
    return;
  }
  if (argv.length >= 2 && argv[0] === '--number-bits') {
    const result = argv.slice(1).map((hex) => {
      const value = numberFromBits(hex);
      try {
        return {ieee754_hex: hex, canonical: canonicalize(value)};
      } catch (error) {
        if (error instanceof OracleError) {
          return {ieee754_hex: hex, error: error.message};
        }
        throw error;
      }
    });
    process.stdout.write(JSON.stringify(result) + '\n');
    return;
  }
  throw new OracleError('usage: phase2b_rfc8785_oracle_v1.mjs --check VECTORS | --number-bits HEX...');
}


try {
  main(process.argv.slice(2));
} catch (error) {
  process.stderr.write(`RFC8785_ORACLE_ERROR: ${error.message}\n`);
  process.exitCode = 1;
}
