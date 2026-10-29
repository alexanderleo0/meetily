//! Keep user-requested inference eligible for CPU while the app is in the
//! background. Do not disable display sleep or system idle sleep.
#[cfg(target_os = "macos")]
pub fn run<T>(work: impl FnOnce() -> T) -> T {
    use objc::{class, msg_send, runtime::Object, sel, sel_impl};
    struct Activity {
        process: *mut Object,
        token: *mut Object,
    }
    impl Drop for Activity {
        fn drop(&mut self) {
            unsafe {
                let _: () = msg_send![self.process, endActivity: self.token];
            }
        }
    }
    objc::rc::autoreleasepool(|| unsafe {
        let process: *mut Object = msg_send![class!(NSProcessInfo), processInfo];
        let reason: *mut Object = msg_send![class!(NSString), alloc];
        let reason: *mut Object = msg_send![reason, initWithUTF8String: b"Meetily local speech recognition\0".as_ptr().cast::<std::ffi::c_char>()];
        // NSActivityUserInitiatedAllowingIdleSystemSleep, from NSProcessInfo.h.
        let token: *mut Object =
            msg_send![process, beginActivityWithOptions: 0x00EFFFFFu64 reason: reason];
        let _: () = msg_send![reason, release];
        let _activity = Activity { process, token };
        work()
    })
}
#[cfg(not(target_os = "macos"))]
pub fn run<T>(work: impl FnOnce() -> T) -> T {
    work()
}
