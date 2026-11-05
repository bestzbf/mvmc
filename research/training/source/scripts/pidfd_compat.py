"""Identity-preserving Linux pidfds, including older Python build headers.

Prefer Python bindings, then exported libc functions. The last fallback is
restricted to the Linux x86_64 LP64 ABI confirmed against the deployment host's
asm/unistd_64.h: pidfd_open=434, pidfd_send_signal=424. Never fall back to a PID
or process-group kill. Capability probing targets only our own pidfd/signal 0.
"""
import ctypes
import os
import platform
import signal


def _libc():
    return ctypes.CDLL(None, use_errno=True)


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _backend(name):
    python = getattr(os if name == 'pidfd_open' else signal, name, None)
    if callable(python):
        return 'python', python
    if platform.system() != 'Linux':
        raise ValueError('pidfd compatibility requires Linux')
    libc = _libc()
    function = getattr(libc, name, None)
    if function is not None:
        function.argtypes = ([ctypes.c_int, ctypes.c_uint] if name == 'pidfd_open' else
                             [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint])
        function.restype = ctypes.c_int
        return 'libc', function
    if platform.machine() != 'x86_64' or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise ValueError('no confirmed pidfd syscall ABI for this architecture')
    function = libc.syscall
    function.restype = ctypes.c_long
    return 'syscall_x86_64', function


def _checked(result):
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(result)


def pidfd_open(pid):
    """Open one positive PID as a close-on-exec kernel process descriptor."""
    pid = _integer(pid, 'pid', 1)
    backend, function = _backend('pidfd_open')
    if backend == 'python':
        return function(pid)
    if backend == 'libc':
        return _checked(function(pid, 0))
    return _checked(function(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0)))


def pidfd_send_signal(fd, signum):
    """Signal only the process already bound to fd; 0 performs a permission probe."""
    fd = _integer(fd, 'pidfd', 0)
    signum = _integer(signum, 'signal', 0)
    backend, function = _backend('pidfd_send_signal')
    if backend == 'python':
        return function(fd, signum)
    if backend == 'libc':
        _checked(function(fd, signum, None, 0))
    else:
        _checked(function(ctypes.c_long(424), ctypes.c_int(fd), ctypes.c_int(signum),
                          ctypes.c_void_p(), ctypes.c_uint(0)))


def probe_pidfd_support():
    """Verify the actual kernel/permission path using our own process, without a signal."""
    open_backend, _ = _backend('pidfd_open')
    signal_backend, _ = _backend('pidfd_send_signal')
    descriptor = pidfd_open(os.getpid())
    try:
        pidfd_send_signal(descriptor, 0)
    finally:
        os.close(descriptor)
    return dict(supported=True, probe_pid=os.getpid(), probe_signal=0,
                open_backend=open_backend, signal_backend=signal_backend,
                system=platform.system(), machine=platform.machine(), kernel=platform.release())
