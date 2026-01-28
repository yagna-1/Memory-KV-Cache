use std::sync::mpsc::{self, Receiver};

use crate::memory_monitor::MemoryPressureLevel;

#[derive(Debug)]
pub struct MemoryPressureHandle {
    receiver: Receiver<MemoryPressureLevel>,
    #[cfg(target_os = "macos")]
    source: *mut std::ffi::c_void,
}

impl MemoryPressureHandle {
    pub fn receiver(&self) -> &Receiver<MemoryPressureLevel> {
        &self.receiver
    }

    pub fn stop(&self) {
        #[cfg(target_os = "macos")]
        unsafe {
            dispatch::cancel_source(self.source);
        }
    }
}

pub fn start_memory_pressure_listener() -> Result<MemoryPressureHandle, String> {
    #[cfg(target_os = "macos")]
    {
        dispatch::start_listener()
    }

    #[cfg(not(target_os = "macos"))]
    {
        Err("Memory pressure events are only supported on macOS".to_string())
    }
}

#[cfg(target_os = "macos")]
mod dispatch {
    use super::{mpsc, MemoryPressureHandle, MemoryPressureLevel};
    use std::ffi::c_void;
    use std::ptr;

    type DispatchSource = *mut c_void;
    type DispatchQueue = *mut c_void;
    type DispatchSourceType = *mut c_void;
    type DispatchFunction = extern "C" fn(*mut c_void);

    const DISPATCH_MEMORYPRESSURE_NORMAL: u64 = 0x1;
    const DISPATCH_MEMORYPRESSURE_WARNING: u64 = 0x2;
    const DISPATCH_MEMORYPRESSURE_CRITICAL: u64 = 0x4;

    #[repr(C)]
    struct PressureContext {
        sender: mpsc::Sender<MemoryPressureLevel>,
        source: DispatchSource,
    }

    #[link(name = "dispatch")]
    extern "C" {
        static _dispatch_source_type_memorypressure: DispatchSourceType;
        fn dispatch_source_create(
            source_type: DispatchSourceType,
            handle: usize,
            mask: u64,
            queue: DispatchQueue,
        ) -> DispatchSource;
        fn dispatch_source_set_event_handler_f(source: DispatchSource, handler: DispatchFunction);
        fn dispatch_source_set_cancel_handler_f(source: DispatchSource, handler: DispatchFunction);
        fn dispatch_source_get_data(source: DispatchSource) -> usize;
        fn dispatch_set_context(object: DispatchSource, context: *mut c_void);
        fn dispatch_queue_create(label: *const i8, attr: *const c_void) -> DispatchQueue;
        fn dispatch_resume(object: *mut c_void);
        fn dispatch_source_cancel(source: DispatchSource);
    }

    pub(super) unsafe fn cancel_source(source: DispatchSource) {
        dispatch_source_cancel(source);
    }

    pub fn start_listener() -> Result<MemoryPressureHandle, String> {
        let (tx, rx) = mpsc::channel();
        let label = b"llm-manager.memory-pressure\0";
        let queue = unsafe { dispatch_queue_create(label.as_ptr() as *const i8, ptr::null()) };
        if queue.is_null() {
            return Err("Failed to create dispatch queue".to_string());
        }

        let mask = DISPATCH_MEMORYPRESSURE_NORMAL
            | DISPATCH_MEMORYPRESSURE_WARNING
            | DISPATCH_MEMORYPRESSURE_CRITICAL;
        let source = unsafe {
            dispatch_source_create(_dispatch_source_type_memorypressure, 0, mask, queue)
        };
        if source.is_null() {
            return Err("Failed to create memory pressure source".to_string());
        }

        let context = Box::new(PressureContext {
            sender: tx,
            source,
        });
        let context_ptr = Box::into_raw(context) as *mut c_void;
        unsafe {
            dispatch_set_context(source, context_ptr);
            dispatch_source_set_event_handler_f(source, pressure_event_handler);
            dispatch_source_set_cancel_handler_f(source, pressure_cancel_handler);
            dispatch_resume(source as *mut c_void);
        }

        Ok(MemoryPressureHandle {
            receiver: rx,
            source,
        })
    }

    extern "C" fn pressure_event_handler(context: *mut c_void) {
        let ctx = unsafe { &*(context as *const PressureContext) };
        let data = unsafe { dispatch_source_get_data(ctx.source) as u64 };
        let level = if data & DISPATCH_MEMORYPRESSURE_CRITICAL != 0 {
            MemoryPressureLevel::Critical
        } else if data & DISPATCH_MEMORYPRESSURE_WARNING != 0 {
            MemoryPressureLevel::Warning
        } else {
            MemoryPressureLevel::Normal
        };
        let _ = ctx.sender.send(level);
    }

    extern "C" fn pressure_cancel_handler(context: *mut c_void) {
        unsafe {
            let _ = Box::from_raw(context as *mut PressureContext);
        }
    }
}
