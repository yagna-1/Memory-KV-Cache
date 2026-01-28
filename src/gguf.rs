use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug, Clone)]
pub enum GgufValue {
    Bool(bool),
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
    F64(f64),
    String(String),
    Array { element_type: u32, count: u64 },
}

impl GgufValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U8(v) => Some(*v as u64),
            GgufValue::U16(v) => Some(*v as u64),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I8(v) => (*v >= 0).then_some(*v as u64),
            GgufValue::I16(v) => (*v >= 0).then_some(*v as u64),
            GgufValue::I32(v) => (*v >= 0).then_some(*v as u64),
            GgufValue::I64(v) => (*v >= 0).then_some(*v as u64),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub version: u32,
    pub kv: HashMap<String, GgufValue>,
}

pub fn read_metadata(path: &Path) -> Result<GgufMetadata, String> {
    let mut file = File::open(path)
        .map_err(|err| format!("Failed to open {}: {err}", path.display()))?;
    let mut header = [0u8; 4];
    file.read_exact(&mut header)
        .map_err(|err| format!("Failed to read header: {err}"))?;
    if &header != b"GGUF" {
        return Err("Invalid GGUF magic header".to_string());
    }

    let version = read_u32(&mut file)?;
    let _tensor_count = read_u64(&mut file)?;
    let kv_count = read_u64(&mut file)?;

    let mut kv = HashMap::new();
    for _ in 0..kv_count {
        let key = read_string(&mut file)?;
        let value_type = read_u32(&mut file)?;
        let value = read_value(&mut file, value_type)?;
        kv.insert(key, value);
    }

    Ok(GgufMetadata { version, kv })
}

fn read_value(file: &mut File, value_type: u32) -> Result<GgufValue, String> {
    match value_type {
        0 => Ok(GgufValue::U8(read_u8(file)?)),
        1 => Ok(GgufValue::I8(read_i8(file)?)),
        2 => Ok(GgufValue::U16(read_u16(file)?)),
        3 => Ok(GgufValue::I16(read_i16(file)?)),
        4 => Ok(GgufValue::U32(read_u32(file)?)),
        5 => Ok(GgufValue::I32(read_i32(file)?)),
        6 => Ok(GgufValue::F32(read_f32(file)?)),
        7 => Ok(GgufValue::Bool(read_u8(file)? != 0)),
        8 => Ok(GgufValue::String(read_string(file)?)),
        9 => {
            let element_type = read_u32(file)?;
            let count = read_u64(file)?;
            skip_array(file, element_type, count)?;
            Ok(GgufValue::Array {
                element_type,
                count,
            })
        }
        10 => Ok(GgufValue::U64(read_u64(file)?)),
        11 => Ok(GgufValue::I64(read_i64(file)?)),
        12 => Ok(GgufValue::F64(read_f64(file)?)),
        other => Err(format!("Unsupported GGUF value type: {other}")),
    }
}

fn skip_array(file: &mut File, element_type: u32, count: u64) -> Result<(), String> {
    match element_type {
        0 => skip_bytes(file, count as u64)?,
        1 => skip_bytes(file, count as u64)?,
        2 => skip_bytes(file, count * 2)?,
        3 => skip_bytes(file, count * 2)?,
        4 => skip_bytes(file, count * 4)?,
        5 => skip_bytes(file, count * 4)?,
        6 => skip_bytes(file, count * 4)?,
        7 => skip_bytes(file, count as u64)?,
        8 => {
            for _ in 0..count {
                let _ = read_string(file)?;
            }
        }
        9 => return Err("Nested arrays in GGUF are not supported".to_string()),
        10 => skip_bytes(file, count * 8)?,
        11 => skip_bytes(file, count * 8)?,
        12 => skip_bytes(file, count * 8)?,
        other => return Err(format!("Unsupported GGUF array element type: {other}")),
    }
    Ok(())
}

fn skip_bytes(file: &mut File, count: u64) -> Result<(), String> {
    file.seek(SeekFrom::Current(count as i64))
        .map_err(|err| format!("Failed to skip bytes: {err}"))?;
    Ok(())
}

fn read_u8(file: &mut File) -> Result<u8, String> {
    let mut buf = [0u8; 1];
    file.read_exact(&mut buf)
        .map_err(|err| format!("Failed to read u8: {err}"))?;
    Ok(buf[0])
}

fn read_i8(file: &mut File) -> Result<i8, String> {
    Ok(read_u8(file)? as i8)
}

fn read_u16(file: &mut File) -> Result<u16, String> {
    let mut buf = [0u8; 2];
    file.read_exact(&mut buf)
        .map_err(|err| format!("Failed to read u16: {err}"))?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16(file: &mut File) -> Result<i16, String> {
    Ok(read_u16(file)? as i16)
}

fn read_u32(file: &mut File) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)
        .map_err(|err| format!("Failed to read u32: {err}"))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(file: &mut File) -> Result<i32, String> {
    Ok(read_u32(file)? as i32)
}

fn read_u64(file: &mut File) -> Result<u64, String> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf)
        .map_err(|err| format!("Failed to read u64: {err}"))?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64(file: &mut File) -> Result<i64, String> {
    Ok(read_u64(file)? as i64)
}

fn read_f32(file: &mut File) -> Result<f32, String> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)
        .map_err(|err| format!("Failed to read f32: {err}"))?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64(file: &mut File) -> Result<f64, String> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf)
        .map_err(|err| format!("Failed to read f64: {err}"))?;
    Ok(f64::from_le_bytes(buf))
}

fn read_string(file: &mut File) -> Result<String, String> {
    let len = read_u32(file)? as usize;
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf)
        .map_err(|err| format!("Failed to read string: {err}"))?;
    String::from_utf8(buf).map_err(|err| format!("Invalid UTF-8 string: {err}"))
}

#[cfg(test)]
mod tests {
    use super::{read_metadata, GgufValue};
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;

    fn write_u32(file: &mut File, value: u32) {
        file.write_all(&value.to_le_bytes()).unwrap();
    }

    fn write_u64(file: &mut File, value: u64) {
        file.write_all(&value.to_le_bytes()).unwrap();
    }

    fn write_string(file: &mut File, value: &str) {
        write_u32(file, value.len() as u32);
        file.write_all(value.as_bytes()).unwrap();
    }

    #[test]
    fn read_minimal_metadata() {
        let mut path = PathBuf::from(std::env::temp_dir());
        path.push("test_minimal.gguf");
        let mut file = File::create(&path).unwrap();

        file.write_all(b"GGUF").unwrap();
        write_u32(&mut file, 2); // version
        write_u64(&mut file, 0); // tensor count
        write_u64(&mut file, 3); // kv count

        write_string(&mut file, "llama.n_layer");
        write_u32(&mut file, 4); // U32
        write_u32(&mut file, 32);

        write_string(&mut file, "llama.hidden_size");
        write_u32(&mut file, 4); // U32
        write_u32(&mut file, 4096);

        write_string(&mut file, "general.architecture");
        write_u32(&mut file, 8); // string
        write_string(&mut file, "llama");

        drop(file);

        let metadata = read_metadata(&path).unwrap();
        assert_eq!(metadata.version, 2);
        assert!(matches!(
            metadata.kv.get("llama.n_layer"),
            Some(GgufValue::U32(32))
        ));
        assert!(matches!(
            metadata.kv.get("llama.hidden_size"),
            Some(GgufValue::U32(4096))
        ));
        assert!(matches!(
            metadata.kv.get("general.architecture"),
            Some(GgufValue::String(value)) if value == "llama"
        ));

        let _ = std::fs::remove_file(path);
    }
}
