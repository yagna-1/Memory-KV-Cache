use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver},
    Arc, Mutex,
};
use std::process::Command;
use std::thread;
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone, Copy)]
pub enum MemoryPressureLevel {
    Normal,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    pub resident_bytes: u64,
    pub virtual_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct MemoryEvent {
    pub level: MemoryPressureLevel,
    pub stats: MemoryStats,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryThresholds {
    pub warning_bytes: u64,
    pub critical_bytes: u64,
}

impl MemoryThresholds {
    pub fn evaluate(&self, stats: MemoryStats) -> MemoryPressureLevel {
        if stats.resident_bytes >= self.critical_bytes {
            MemoryPressureLevel::Critical
        } else if stats.resident_bytes >= self.warning_bytes {
            MemoryPressureLevel::Warning
        } else {
            MemoryPressureLevel::Normal
        }
    }
}

pub struct MemoryMonitor {
    poll_interval: Duration,
    thresholds: MemoryThresholds,
    pid: Option<u32>,
    history_limit: usize,
}

pub struct MemoryMonitorHandle {
    stop_flag: Arc<AtomicBool>,
    receiver: Receiver<MemoryEvent>,
    history: Arc<Mutex<Vec<MemoryEvent>>>,
}

#[derive(Clone, Default)]
pub struct MemoryCallbacks {
    on_normal: Option<Arc<dyn Fn(MemoryEvent) + Send + Sync>>,
    on_warning: Option<Arc<dyn Fn(MemoryEvent) + Send + Sync>>,
    on_critical: Option<Arc<dyn Fn(MemoryEvent) + Send + Sync>>,
}

impl MemoryCallbacks {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn on_normal<F: Fn(MemoryEvent) + Send + Sync + 'static>(mut self, callback: F) -> Self {
        self.on_normal = Some(Arc::new(callback));
        self
    }

    pub fn on_warning<F: Fn(MemoryEvent) + Send + Sync + 'static>(mut self, callback: F) -> Self {
        self.on_warning = Some(Arc::new(callback));
        self
    }

    pub fn on_critical<F: Fn(MemoryEvent) + Send + Sync + 'static>(mut self, callback: F) -> Self {
        self.on_critical = Some(Arc::new(callback));
        self
    }

    fn trigger(&self, event: &MemoryEvent) {
        match event.level {
            MemoryPressureLevel::Normal => {
                if let Some(cb) = &self.on_normal {
                    cb(event.clone());
                }
            }
            MemoryPressureLevel::Warning => {
                if let Some(cb) = &self.on_warning {
                    cb(event.clone());
                }
            }
            MemoryPressureLevel::Critical => {
                if let Some(cb) = &self.on_critical {
                    cb(event.clone());
                }
            }
        }
    }
}

impl MemoryMonitor {
    pub fn new(poll_interval: Duration, thresholds: MemoryThresholds) -> Self {
        Self {
            poll_interval,
            thresholds,
            pid: None,
            history_limit: 512,
        }
    }

    pub fn with_pid(mut self, pid: u32) -> Self {
        self.pid = Some(pid);
        self
    }

    pub fn with_history_limit(mut self, limit: usize) -> Self {
        self.history_limit = limit.max(1);
        self
    }

    pub fn start(&self) -> MemoryMonitorHandle {
        self.start_with_callbacks(MemoryCallbacks::default())
    }

    pub fn start_with_callbacks(&self, callbacks: MemoryCallbacks) -> MemoryMonitorHandle {
        let (tx, rx) = mpsc::channel();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_thread = stop_flag.clone();
        let thresholds = self.thresholds;
        let poll_interval = self.poll_interval;
        let pid = self.pid.unwrap_or_else(std::process::id);
        let history_limit = self.history_limit;
        let history = Arc::new(Mutex::new(Vec::new()));
        let history_thread = history.clone();

        thread::spawn(move || {
            while !stop_thread.load(Ordering::SeqCst) {
                let stats = process_memory(pid).unwrap_or(MemoryStats {
                    resident_bytes: 0,
                    virtual_bytes: 0,
                });
                let level = thresholds.evaluate(stats);
                let event = MemoryEvent {
                    level,
                    stats,
                    timestamp: SystemTime::now(),
                };
                callbacks.trigger(&event);
                if let Ok(mut entries) = history_thread.lock() {
                    entries.push(event.clone());
                    if entries.len() > history_limit {
                        let overflow = entries.len() - history_limit;
                        entries.drain(0..overflow);
                    }
                }
                let _ = tx.send(event);
                thread::sleep(poll_interval);
            }
        });

        MemoryMonitorHandle {
            stop_flag,
            receiver: rx,
            history,
        }
    }
}

impl MemoryMonitorHandle {
    pub fn receiver(&self) -> &Receiver<MemoryEvent> {
        &self.receiver
    }

    pub fn history_snapshot(&self) -> Vec<MemoryEvent> {
        self.history
            .lock()
            .map(|entries| entries.clone())
            .unwrap_or_default()
    }

    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::SeqCst);
    }
}

fn current_process_memory() -> Option<MemoryStats> {
    process_memory(std::process::id())
}

#[cfg(target_os = "macos")]
fn process_memory(pid: u32) -> Option<MemoryStats> {
    mach::memory_for_pid(pid).or_else(|| process_memory_ps(pid))
}

#[cfg(not(target_os = "macos"))]
fn process_memory(pid: u32) -> Option<MemoryStats> {
    process_memory_ps(pid)
}

fn process_memory_ps(pid: u32) -> Option<MemoryStats> {
    let output = Command::new("ps")
        .args(["-o", "rss=", "-o", "vsz=", "-p", &pid.to_string()])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut parts = stdout.split_whitespace();
    let rss_kb = parts.next()?.parse::<u64>().ok()?;
    let vsz_kb = parts.next()?.parse::<u64>().ok()?;

    Some(MemoryStats {
        resident_bytes: rss_kb * 1024,
        virtual_bytes: vsz_kb * 1024,
    })
}

#[cfg(target_os = "macos")]
mod mach {
    use super::MemoryStats;
    use std::mem::{size_of, MaybeUninit};

    type KernReturn = i32;
    type MachPort = u32;
    type MachMsgTypeNumber = u32;
    type Integer = i32;
    type Natural = u32;

    const KERN_SUCCESS: KernReturn = 0;
    const TASK_BASIC_INFO_64: i32 = 5;

    #[repr(C)]
    #[derive(Default, Copy, Clone)]
    struct TimeValue {
        seconds: i32,
        microseconds: i32,
    }

    #[repr(C)]
    #[derive(Default, Copy, Clone)]
    struct TaskBasicInfo64 {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: TimeValue,
        system_time: TimeValue,
        policy: i32,
        suspend_count: i32,
    }

    #[link(name = "System")]
    extern "C" {
        static mach_task_self_: MachPort;
        fn task_info(
            target_task: MachPort,
            flavor: i32,
            task_info_out: *mut Integer,
            task_info_out_cnt: *mut MachMsgTypeNumber,
        ) -> KernReturn;
        fn task_for_pid(target_tport: MachPort, pid: i32, task: *mut MachPort) -> KernReturn;
    }

    pub fn memory_for_pid(pid: u32) -> Option<MemoryStats> {
        let task = if pid == std::process::id() {
            unsafe { mach_task_self_ }
        } else {
            let mut task: MachPort = 0;
            let kr = unsafe { task_for_pid(unsafe { mach_task_self_ }, pid as i32, &mut task) };
            if kr != KERN_SUCCESS {
                return None;
            }
            task
        };

        let mut info = MaybeUninit::<TaskBasicInfo64>::zeroed();
        let mut count: MachMsgTypeNumber =
            (size_of::<TaskBasicInfo64>() / size_of::<Natural>()) as MachMsgTypeNumber;

        let kr = unsafe {
            task_info(
                task,
                TASK_BASIC_INFO_64,
                info.as_mut_ptr() as *mut Integer,
                &mut count,
            )
        };
        if kr != KERN_SUCCESS {
            return None;
        }

        let info = unsafe { info.assume_init() };
        Some(MemoryStats {
            resident_bytes: info.resident_size,
            virtual_bytes: info.virtual_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{MemoryCallbacks, MemoryPressureLevel, MemoryStats, MemoryThresholds};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn thresholds_evaluate_levels() {
        let thresholds = MemoryThresholds {
            warning_bytes: 100,
            critical_bytes: 200,
        };
        let normal = thresholds.evaluate(MemoryStats {
            resident_bytes: 50,
            virtual_bytes: 0,
        });
        let warning = thresholds.evaluate(MemoryStats {
            resident_bytes: 150,
            virtual_bytes: 0,
        });
        let critical = thresholds.evaluate(MemoryStats {
            resident_bytes: 250,
            virtual_bytes: 0,
        });

        assert!(matches!(normal, MemoryPressureLevel::Normal));
        assert!(matches!(warning, MemoryPressureLevel::Warning));
        assert!(matches!(critical, MemoryPressureLevel::Critical));
    }

    #[test]
    fn history_limit_is_enforced() {
        let thresholds = MemoryThresholds {
            warning_bytes: 0,
            critical_bytes: u64::MAX,
        };
        let monitor = super::MemoryMonitor::new(Duration::from_millis(10), thresholds)
            .with_history_limit(3);
        let handle = monitor.start();
        std::thread::sleep(Duration::from_millis(50));
        handle.stop();

        let history = handle.history_snapshot();
        assert!(!history.is_empty());
        assert!(history.len() <= 3);
    }

    #[test]
    fn callbacks_fire_for_warning() {
        let thresholds = MemoryThresholds {
            warning_bytes: 0,
            critical_bytes: u64::MAX,
        };
        let counter = Arc::new(AtomicUsize::new(0));
        let callbacks = MemoryCallbacks::new().on_warning({
            let counter = counter.clone();
            move |_| {
                counter.fetch_add(1, Ordering::SeqCst);
            }
        });
        let monitor =
            super::MemoryMonitor::new(Duration::from_millis(10), thresholds).with_history_limit(2);
        let handle = monitor.start_with_callbacks(callbacks);
        std::thread::sleep(Duration::from_millis(50));
        handle.stop();
        assert!(counter.load(Ordering::SeqCst) > 0);
    }
}
