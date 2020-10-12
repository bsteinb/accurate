(function() {var implementors = {};
implementors["accurate"] = [{"text":"impl&lt;F&gt; Sync for Dot2&lt;F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;F, R&gt; Sync for DotK&lt;F, R&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;R: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;F&gt; Sync for NaiveDot&lt;F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;F&gt; Sync for OnlineExactDot&lt;F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;Acc&gt; Sync for DotFolder&lt;Acc&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;Acc: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;Acc&gt; Sync for DotConsumer&lt;Acc&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;Acc: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;F&gt; Sync for NaiveSum&lt;F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;F&gt; Sync for OnlineExactSum&lt;F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;F&gt; Sync for Sum2&lt;F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;F, C&gt; Sync for SumK&lt;F, C&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;C: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;Acc&gt; Sync for SumFolder&lt;Acc&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;Acc: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;Acc&gt; Sync for SumConsumer&lt;Acc&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;Acc: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl Sync for AddReducer","synthetic":true,"types":[]}];
implementors["crossbeam_channel"] = [{"text":"impl&lt;T&gt; Sync for IntoIter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Send,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; Sync for Iter&lt;'a, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Send,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; Sync for TryIter&lt;'a, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Send,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a&gt; !Sync for SelectedOperation&lt;'a&gt;","synthetic":true,"types":[]},{"text":"impl Sync for ReadyTimeoutError","synthetic":true,"types":[]},{"text":"impl Sync for SelectTimeoutError","synthetic":true,"types":[]},{"text":"impl Sync for TryReadyError","synthetic":true,"types":[]},{"text":"impl Sync for TrySelectError","synthetic":true,"types":[]},{"text":"impl Sync for RecvError","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; Sync for SendError&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl Sync for RecvTimeoutError","synthetic":true,"types":[]},{"text":"impl Sync for TryRecvError","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; Sync for SendTimeoutError&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; Sync for TrySendError&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T:&nbsp;Send&gt; Sync for Sender&lt;T&gt;","synthetic":false,"types":[]},{"text":"impl&lt;T:&nbsp;Send&gt; Sync for Receiver&lt;T&gt;","synthetic":false,"types":[]},{"text":"impl&lt;'a&gt; Sync for Select&lt;'a&gt;","synthetic":false,"types":[]}];
implementors["crossbeam_deque"] = [{"text":"impl&lt;T&gt; !Sync for Worker&lt;T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; Sync for Steal&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T:&nbsp;Send&gt; Sync for Stealer&lt;T&gt;","synthetic":false,"types":[]},{"text":"impl&lt;T:&nbsp;Send&gt; Sync for Injector&lt;T&gt;","synthetic":false,"types":[]}];
implementors["crossbeam_epoch"] = [{"text":"impl&lt;'g, T, P&gt; !Sync for CompareAndSetError&lt;'g, T, P&gt;","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; Sync for Owned&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'g, T&gt; !Sync for Shared&lt;'g, T&gt;","synthetic":true,"types":[]},{"text":"impl !Sync for LocalHandle","synthetic":true,"types":[]},{"text":"impl !Sync for Guard","synthetic":true,"types":[]},{"text":"impl&lt;T:&nbsp;Send + Sync&gt; Sync for Atomic&lt;T&gt;","synthetic":false,"types":[]},{"text":"impl Sync for Collector","synthetic":false,"types":[]}];
implementors["crossbeam_utils"] = [{"text":"impl !Sync for Backoff","synthetic":true,"types":[]},{"text":"impl !Sync for Parker","synthetic":true,"types":[]},{"text":"impl Sync for WaitGroup","synthetic":true,"types":[]},{"text":"impl&lt;'scope, 'env&gt; Sync for ScopedThreadBuilder&lt;'scope, 'env&gt;","synthetic":true,"types":[]},{"text":"impl&lt;T:&nbsp;Send&gt; Sync for AtomicCell&lt;T&gt;","synthetic":false,"types":[]},{"text":"impl&lt;T:&nbsp;Sync&gt; Sync for CachePadded&lt;T&gt;","synthetic":false,"types":[]},{"text":"impl Sync for Unparker","synthetic":false,"types":[]},{"text":"impl&lt;T:&nbsp;?Sized + Send + Sync&gt; Sync for ShardedLock&lt;T&gt;","synthetic":false,"types":[]},{"text":"impl&lt;'a, T:&nbsp;?Sized + Sync&gt; Sync for ShardedLockReadGuard&lt;'a, T&gt;","synthetic":false,"types":[]},{"text":"impl&lt;'a, T:&nbsp;?Sized + Sync&gt; Sync for ShardedLockWriteGuard&lt;'a, T&gt;","synthetic":false,"types":[]},{"text":"impl&lt;'env&gt; Sync for Scope&lt;'env&gt;","synthetic":false,"types":[]},{"text":"impl&lt;'scope, T&gt; Sync for ScopedJoinHandle&lt;'scope, T&gt;","synthetic":false,"types":[]}];
implementors["either"] = [{"text":"impl&lt;L, R&gt; Sync for Either&lt;L, R&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;L: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;R: Sync,&nbsp;</span>","synthetic":true,"types":[]}];
implementors["ieee754"] = [{"text":"impl&lt;T&gt; Sync for Iter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]}];
implementors["libc"] = [{"text":"impl !Sync for group","synthetic":true,"types":[]},{"text":"impl Sync for utimbuf","synthetic":true,"types":[]},{"text":"impl Sync for timeval","synthetic":true,"types":[]},{"text":"impl Sync for timespec","synthetic":true,"types":[]},{"text":"impl Sync for rlimit","synthetic":true,"types":[]},{"text":"impl Sync for rusage","synthetic":true,"types":[]},{"text":"impl Sync for ipv6_mreq","synthetic":true,"types":[]},{"text":"impl !Sync for hostent","synthetic":true,"types":[]},{"text":"impl !Sync for iovec","synthetic":true,"types":[]},{"text":"impl Sync for pollfd","synthetic":true,"types":[]},{"text":"impl Sync for winsize","synthetic":true,"types":[]},{"text":"impl Sync for linger","synthetic":true,"types":[]},{"text":"impl !Sync for sigval","synthetic":true,"types":[]},{"text":"impl Sync for itimerval","synthetic":true,"types":[]},{"text":"impl Sync for tms","synthetic":true,"types":[]},{"text":"impl !Sync for servent","synthetic":true,"types":[]},{"text":"impl !Sync for protoent","synthetic":true,"types":[]},{"text":"impl Sync for in_addr","synthetic":true,"types":[]},{"text":"impl Sync for ip_mreq","synthetic":true,"types":[]},{"text":"impl Sync for ip_mreq_source","synthetic":true,"types":[]},{"text":"impl Sync for sockaddr","synthetic":true,"types":[]},{"text":"impl Sync for sockaddr_in","synthetic":true,"types":[]},{"text":"impl Sync for sockaddr_in6","synthetic":true,"types":[]},{"text":"impl !Sync for addrinfo","synthetic":true,"types":[]},{"text":"impl Sync for sockaddr_ll","synthetic":true,"types":[]},{"text":"impl Sync for fd_set","synthetic":true,"types":[]},{"text":"impl !Sync for tm","synthetic":true,"types":[]},{"text":"impl Sync for sched_param","synthetic":true,"types":[]},{"text":"impl !Sync for Dl_info","synthetic":true,"types":[]},{"text":"impl !Sync for lconv","synthetic":true,"types":[]},{"text":"impl Sync for in_pktinfo","synthetic":true,"types":[]},{"text":"impl !Sync for ifaddrs","synthetic":true,"types":[]},{"text":"impl Sync for in6_rtmsg","synthetic":true,"types":[]},{"text":"impl Sync for arpreq","synthetic":true,"types":[]},{"text":"impl Sync for arpreq_old","synthetic":true,"types":[]},{"text":"impl Sync for arphdr","synthetic":true,"types":[]},{"text":"impl !Sync for mmsghdr","synthetic":true,"types":[]},{"text":"impl Sync for epoll_event","synthetic":true,"types":[]},{"text":"impl Sync for sockaddr_un","synthetic":true,"types":[]},{"text":"impl Sync for sockaddr_storage","synthetic":true,"types":[]},{"text":"impl Sync for utsname","synthetic":true,"types":[]},{"text":"impl !Sync for sigevent","synthetic":true,"types":[]},{"text":"impl Sync for rlimit64","synthetic":true,"types":[]},{"text":"impl !Sync for glob_t","synthetic":true,"types":[]},{"text":"impl !Sync for passwd","synthetic":true,"types":[]},{"text":"impl !Sync for spwd","synthetic":true,"types":[]},{"text":"impl Sync for dqblk","synthetic":true,"types":[]},{"text":"impl Sync for signalfd_siginfo","synthetic":true,"types":[]},{"text":"impl Sync for itimerspec","synthetic":true,"types":[]},{"text":"impl Sync for fsid_t","synthetic":true,"types":[]},{"text":"impl Sync for packet_mreq","synthetic":true,"types":[]},{"text":"impl Sync for cpu_set_t","synthetic":true,"types":[]},{"text":"impl !Sync for if_nameindex","synthetic":true,"types":[]},{"text":"impl Sync for msginfo","synthetic":true,"types":[]},{"text":"impl Sync for sembuf","synthetic":true,"types":[]},{"text":"impl Sync for input_event","synthetic":true,"types":[]},{"text":"impl Sync for input_id","synthetic":true,"types":[]},{"text":"impl Sync for input_absinfo","synthetic":true,"types":[]},{"text":"impl Sync for input_keymap_entry","synthetic":true,"types":[]},{"text":"impl Sync for input_mask","synthetic":true,"types":[]},{"text":"impl Sync for ff_replay","synthetic":true,"types":[]},{"text":"impl Sync for ff_trigger","synthetic":true,"types":[]},{"text":"impl Sync for ff_envelope","synthetic":true,"types":[]},{"text":"impl Sync for ff_constant_effect","synthetic":true,"types":[]},{"text":"impl Sync for ff_ramp_effect","synthetic":true,"types":[]},{"text":"impl Sync for ff_condition_effect","synthetic":true,"types":[]},{"text":"impl !Sync for ff_periodic_effect","synthetic":true,"types":[]},{"text":"impl Sync for ff_rumble_effect","synthetic":true,"types":[]},{"text":"impl Sync for ff_effect","synthetic":true,"types":[]},{"text":"impl !Sync for dl_phdr_info","synthetic":true,"types":[]},{"text":"impl Sync for Elf32_Ehdr","synthetic":true,"types":[]},{"text":"impl Sync for Elf64_Ehdr","synthetic":true,"types":[]},{"text":"impl Sync for Elf32_Sym","synthetic":true,"types":[]},{"text":"impl Sync for Elf64_Sym","synthetic":true,"types":[]},{"text":"impl Sync for Elf32_Phdr","synthetic":true,"types":[]},{"text":"impl Sync for Elf64_Phdr","synthetic":true,"types":[]},{"text":"impl Sync for Elf32_Shdr","synthetic":true,"types":[]},{"text":"impl Sync for Elf64_Shdr","synthetic":true,"types":[]},{"text":"impl Sync for Elf32_Chdr","synthetic":true,"types":[]},{"text":"impl Sync for Elf64_Chdr","synthetic":true,"types":[]},{"text":"impl Sync for ucred","synthetic":true,"types":[]},{"text":"impl !Sync for mntent","synthetic":true,"types":[]},{"text":"impl !Sync for posix_spawn_file_actions_t","synthetic":true,"types":[]},{"text":"impl Sync for posix_spawnattr_t","synthetic":true,"types":[]},{"text":"impl Sync for genlmsghdr","synthetic":true,"types":[]},{"text":"impl Sync for in6_pktinfo","synthetic":true,"types":[]},{"text":"impl Sync for arpd_request","synthetic":true,"types":[]},{"text":"impl Sync for inotify_event","synthetic":true,"types":[]},{"text":"impl Sync for fanotify_response","synthetic":true,"types":[]},{"text":"impl Sync for sockaddr_vm","synthetic":true,"types":[]},{"text":"impl Sync for regmatch_t","synthetic":true,"types":[]},{"text":"impl Sync for sock_extended_err","synthetic":true,"types":[]},{"text":"impl Sync for sockaddr_nl","synthetic":true,"types":[]},{"text":"impl Sync for dirent","synthetic":true,"types":[]},{"text":"impl Sync for dirent64","synthetic":true,"types":[]},{"text":"impl Sync for sockaddr_alg","synthetic":true,"types":[]},{"text":"impl Sync for af_alg_iv","synthetic":true,"types":[]},{"text":"impl Sync for mq_attr","synthetic":true,"types":[]},{"text":"impl Sync for statx","synthetic":true,"types":[]},{"text":"impl Sync for statx_timestamp","synthetic":true,"types":[]},{"text":"impl !Sync for aiocb","synthetic":true,"types":[]},{"text":"impl Sync for __exit_status","synthetic":true,"types":[]},{"text":"impl Sync for __timeval","synthetic":true,"types":[]},{"text":"impl !Sync for glob64_t","synthetic":true,"types":[]},{"text":"impl !Sync for msghdr","synthetic":true,"types":[]},{"text":"impl Sync for cmsghdr","synthetic":true,"types":[]},{"text":"impl Sync for termios","synthetic":true,"types":[]},{"text":"impl Sync for mallinfo","synthetic":true,"types":[]},{"text":"impl Sync for nlmsghdr","synthetic":true,"types":[]},{"text":"impl Sync for nlmsgerr","synthetic":true,"types":[]},{"text":"impl Sync for nl_pktinfo","synthetic":true,"types":[]},{"text":"impl Sync for nl_mmap_req","synthetic":true,"types":[]},{"text":"impl Sync for nl_mmap_hdr","synthetic":true,"types":[]},{"text":"impl Sync for nlattr","synthetic":true,"types":[]},{"text":"impl !Sync for rtentry","synthetic":true,"types":[]},{"text":"impl Sync for timex","synthetic":true,"types":[]},{"text":"impl Sync for ntptimeval","synthetic":true,"types":[]},{"text":"impl !Sync for regex_t","synthetic":true,"types":[]},{"text":"impl Sync for utmpx","synthetic":true,"types":[]},{"text":"impl Sync for sigset_t","synthetic":true,"types":[]},{"text":"impl Sync for sysinfo","synthetic":true,"types":[]},{"text":"impl Sync for msqid_ds","synthetic":true,"types":[]},{"text":"impl Sync for sigaction","synthetic":true,"types":[]},{"text":"impl Sync for statfs","synthetic":true,"types":[]},{"text":"impl Sync for flock","synthetic":true,"types":[]},{"text":"impl Sync for flock64","synthetic":true,"types":[]},{"text":"impl Sync for siginfo_t","synthetic":true,"types":[]},{"text":"impl !Sync for stack_t","synthetic":true,"types":[]},{"text":"impl Sync for stat","synthetic":true,"types":[]},{"text":"impl Sync for stat64","synthetic":true,"types":[]},{"text":"impl Sync for statfs64","synthetic":true,"types":[]},{"text":"impl Sync for statvfs64","synthetic":true,"types":[]},{"text":"impl Sync for pthread_attr_t","synthetic":true,"types":[]},{"text":"impl Sync for _libc_fpxreg","synthetic":true,"types":[]},{"text":"impl Sync for _libc_xmmreg","synthetic":true,"types":[]},{"text":"impl Sync for _libc_fpstate","synthetic":true,"types":[]},{"text":"impl Sync for user_regs_struct","synthetic":true,"types":[]},{"text":"impl !Sync for user","synthetic":true,"types":[]},{"text":"impl !Sync for mcontext_t","synthetic":true,"types":[]},{"text":"impl Sync for ipc_perm","synthetic":true,"types":[]},{"text":"impl Sync for shmid_ds","synthetic":true,"types":[]},{"text":"impl Sync for termios2","synthetic":true,"types":[]},{"text":"impl Sync for ip_mreqn","synthetic":true,"types":[]},{"text":"impl Sync for user_fpregs_struct","synthetic":true,"types":[]},{"text":"impl !Sync for ucontext_t","synthetic":true,"types":[]},{"text":"impl Sync for statvfs","synthetic":true,"types":[]},{"text":"impl Sync for max_align_t","synthetic":true,"types":[]},{"text":"impl Sync for sem_t","synthetic":true,"types":[]},{"text":"impl Sync for pthread_mutexattr_t","synthetic":true,"types":[]},{"text":"impl Sync for pthread_rwlockattr_t","synthetic":true,"types":[]},{"text":"impl Sync for pthread_condattr_t","synthetic":true,"types":[]},{"text":"impl Sync for fanotify_event_metadata","synthetic":true,"types":[]},{"text":"impl Sync for pthread_cond_t","synthetic":true,"types":[]},{"text":"impl Sync for pthread_mutex_t","synthetic":true,"types":[]},{"text":"impl Sync for pthread_rwlock_t","synthetic":true,"types":[]},{"text":"impl Sync for in6_addr","synthetic":true,"types":[]},{"text":"impl Sync for DIR","synthetic":true,"types":[]},{"text":"impl Sync for FILE","synthetic":true,"types":[]},{"text":"impl Sync for fpos_t","synthetic":true,"types":[]},{"text":"impl Sync for timezone","synthetic":true,"types":[]},{"text":"impl Sync for fpos64_t","synthetic":true,"types":[]}];
implementors["num_traits"] = [{"text":"impl Sync for ParseFloatError","synthetic":true,"types":[]},{"text":"impl Sync for FloatErrorKind","synthetic":true,"types":[]}];
implementors["rayon"] = [{"text":"impl&lt;T&gt; Sync for IntoIter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; Sync for Iter&lt;'a, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; Sync for Drain&lt;'a, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;K, V&gt; Sync for IntoIter&lt;K, V&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;K: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;V: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, K, V&gt; Sync for Iter&lt;'a, K, V&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'a, K, V&gt; Sync for IterMut&lt;'a, K, V&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;V: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; Sync for IntoIter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; Sync for Iter&lt;'a, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;K, V&gt; Sync for IntoIter&lt;K, V&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;K: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;V: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, K, V&gt; Sync for Iter&lt;'a, K, V&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'a, K, V&gt; Sync for IterMut&lt;'a, K, V&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;V: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, K, V&gt; Sync for Drain&lt;'a, K, V&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;K: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;V: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; Sync for IntoIter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; Sync for Iter&lt;'a, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; Sync for Drain&lt;'a, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; Sync for IntoIter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; Sync for Iter&lt;'a, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; Sync for IterMut&lt;'a, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; Sync for IntoIter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; Sync for Iter&lt;'a, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; Sync for IterMut&lt;'a, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; Sync for Drain&lt;'a, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;A, B&gt; Sync for Chain&lt;A, B&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;A: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;B: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; Sync for Chunks&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; Sync for Cloned&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; Sync for Copied&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; Sync for Empty&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; Sync for Enumerate&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, P&gt; Sync for Filter&lt;I, P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, P&gt; Sync for FilterMap&lt;I, P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, F&gt; Sync for FlatMap&lt;I, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, F&gt; Sync for FlatMapIter&lt;I, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; Sync for Flatten&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; Sync for FlattenIter&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, ID, F&gt; Sync for Fold&lt;I, ID, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;ID: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, U, F&gt; Sync for FoldWith&lt;I, U, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;U: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, F&gt; Sync for Inspect&lt;I, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, J&gt; Sync for Interleave&lt;I, J&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;J: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, J&gt; Sync for InterleaveShortest&lt;I, J&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;J: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; Sync for Intersperse&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;&lt;I as ParallelIterator&gt;::Item: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; Sync for MaxLen&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; Sync for MinLen&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, F&gt; Sync for Map&lt;I, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, INIT, F&gt; Sync for MapInit&lt;I, INIT, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;INIT: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, T, F&gt; Sync for MapWith&lt;I, T, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; Sync for MultiZip&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; Sync for Once&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; Sync for PanicFuse&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;Iter&gt; Sync for IterBridge&lt;Iter&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;Iter: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, P&gt; Sync for Positions&lt;I, P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; Sync for Repeat&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; Sync for RepeatN&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; Sync for Rev&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; Sync for Skip&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;D, S&gt; Sync for Split&lt;D, S&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; Sync for Take&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, U, ID, F&gt; Sync for TryFold&lt;I, U, ID, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;ID: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;U: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, U, F&gt; Sync for TryFoldWith&lt;I, U, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;&lt;U as Try&gt;::Ok: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, F&gt; Sync for Update&lt;I, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; Sync for WhileSome&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;A, B&gt; Sync for Zip&lt;A, B&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;A: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;B: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;A, B&gt; Sync for ZipEq&lt;A, B&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;A: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;B: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; Sync for StepBy&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; Sync for IntoIter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; Sync for Iter&lt;'a, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; Sync for IterMut&lt;'a, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; Sync for Iter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; Sync for Iter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; Sync for IntoIter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; Sync for Iter&lt;'a, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; Sync for IterMut&lt;'a, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'data, T&gt; Sync for Iter&lt;'data, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'data, T&gt; Sync for Chunks&lt;'data, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'data, T&gt; Sync for ChunksExact&lt;'data, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'data, T&gt; Sync for Windows&lt;'data, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'data, T&gt; Sync for IterMut&lt;'data, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'data, T&gt; Sync for ChunksMut&lt;'data, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'data, T&gt; Sync for ChunksExactMut&lt;'data, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'data, T, P&gt; Sync for Split&lt;'data, T, P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'data, T, P&gt; Sync for SplitMut&lt;'data, T, P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;P: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'ch&gt; Sync for Chars&lt;'ch&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'ch&gt; Sync for CharIndices&lt;'ch&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'ch&gt; Sync for Bytes&lt;'ch&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'ch&gt; Sync for EncodeUtf16&lt;'ch&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'ch, P&gt; Sync for Split&lt;'ch, P&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'ch, P&gt; Sync for SplitTerminator&lt;'ch, P&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'ch&gt; Sync for Lines&lt;'ch&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'ch&gt; Sync for SplitWhitespace&lt;'ch&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'ch, P&gt; Sync for Matches&lt;'ch, P&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'ch, P&gt; Sync for MatchIndices&lt;'ch, P&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'a&gt; Sync for Drain&lt;'a&gt;","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; Sync for IntoIter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'data, T&gt; Sync for Drain&lt;'data, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,&nbsp;</span>","synthetic":true,"types":[]}];
implementors["rayon_core"] = [{"text":"impl !Sync for ThreadBuilder","synthetic":true,"types":[]},{"text":"impl&lt;'scope&gt; Sync for Scope&lt;'scope&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'scope&gt; Sync for ScopeFifo&lt;'scope&gt;","synthetic":true,"types":[]},{"text":"impl Sync for ThreadPool","synthetic":true,"types":[]},{"text":"impl Sync for ThreadPoolBuildError","synthetic":true,"types":[]},{"text":"impl&lt;S&nbsp;=&nbsp;DefaultSpawn&gt; !Sync for ThreadPoolBuilder&lt;S&gt;","synthetic":true,"types":[]},{"text":"impl !Sync for Configuration","synthetic":true,"types":[]},{"text":"impl !Sync for FnContext","synthetic":true,"types":[]}];
implementors["scopeguard"] = [{"text":"impl Sync for Always","synthetic":true,"types":[]},{"text":"impl&lt;T, F, S&gt; Sync for ScopeGuard&lt;T, F, S&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Sync,<br>&nbsp;&nbsp;&nbsp;&nbsp;F: FnOnce(T),<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Strategy,&nbsp;</span>","synthetic":false,"types":[]}];
if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()