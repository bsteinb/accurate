(function() {var implementors = {};
implementors["accurate"] = [{"text":"impl&lt;F&gt; UnwindSafe for Dot2&lt;F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;F, R&gt; UnwindSafe for DotK&lt;F, R&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;R: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;F&gt; UnwindSafe for NaiveDot&lt;F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;F&gt; UnwindSafe for OnlineExactDot&lt;F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;Acc&gt; UnwindSafe for DotFolder&lt;Acc&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;Acc: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;Acc&gt; UnwindSafe for DotConsumer&lt;Acc&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;Acc: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;F&gt; UnwindSafe for NaiveSum&lt;F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;F&gt; UnwindSafe for OnlineExactSum&lt;F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;F&gt; UnwindSafe for Sum2&lt;F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;F, C&gt; UnwindSafe for SumK&lt;F, C&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;C: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;F: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;Acc&gt; UnwindSafe for SumFolder&lt;Acc&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;Acc: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;Acc&gt; UnwindSafe for SumConsumer&lt;Acc&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;Acc: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl UnwindSafe for AddReducer","synthetic":true,"types":[]}];
implementors["crossbeam_channel"] = [{"text":"impl&lt;T&gt; UnwindSafe for IntoIter&lt;T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; UnwindSafe for Iter&lt;'a, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; UnwindSafe for TryIter&lt;'a, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'a&gt; !UnwindSafe for Select&lt;'a&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'a&gt; UnwindSafe for SelectedOperation&lt;'a&gt;","synthetic":true,"types":[]},{"text":"impl UnwindSafe for ReadyTimeoutError","synthetic":true,"types":[]},{"text":"impl UnwindSafe for SelectTimeoutError","synthetic":true,"types":[]},{"text":"impl UnwindSafe for TryReadyError","synthetic":true,"types":[]},{"text":"impl UnwindSafe for TrySelectError","synthetic":true,"types":[]},{"text":"impl UnwindSafe for RecvError","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for SendError&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl UnwindSafe for RecvTimeoutError","synthetic":true,"types":[]},{"text":"impl UnwindSafe for TryRecvError","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for SendTimeoutError&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for TrySendError&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for Sender&lt;T&gt;","synthetic":false,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for Receiver&lt;T&gt;","synthetic":false,"types":[]}];
implementors["crossbeam_deque"] = [{"text":"impl&lt;T&gt; UnwindSafe for Worker&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: RefUnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for Stealer&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: RefUnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; !UnwindSafe for Injector&lt;T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for Steal&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]}];
implementors["crossbeam_epoch"] = [{"text":"impl&lt;T&gt; UnwindSafe for Atomic&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: RefUnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'g, T, P&gt; UnwindSafe for CompareAndSetError&lt;'g, T, P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;P: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;T: RefUnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for Owned&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'g, T&gt; UnwindSafe for Shared&lt;'g, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: RefUnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl !UnwindSafe for Collector","synthetic":true,"types":[]},{"text":"impl !UnwindSafe for LocalHandle","synthetic":true,"types":[]},{"text":"impl !UnwindSafe for Guard","synthetic":true,"types":[]}];
implementors["crossbeam_utils"] = [{"text":"impl&lt;T&gt; UnwindSafe for CachePadded&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl UnwindSafe for Backoff","synthetic":true,"types":[]},{"text":"impl !UnwindSafe for Parker","synthetic":true,"types":[]},{"text":"impl !UnwindSafe for Unparker","synthetic":true,"types":[]},{"text":"impl&lt;'a, T:&nbsp;?Sized&gt; UnwindSafe for ShardedLockReadGuard&lt;'a, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'a, T:&nbsp;?Sized&gt; UnwindSafe for ShardedLockWriteGuard&lt;'a, T&gt;","synthetic":true,"types":[]},{"text":"impl !UnwindSafe for WaitGroup","synthetic":true,"types":[]},{"text":"impl&lt;'env&gt; !UnwindSafe for Scope&lt;'env&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'scope, 'env&gt; !UnwindSafe for ScopedThreadBuilder&lt;'scope, 'env&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'scope, T&gt; !UnwindSafe for ScopedJoinHandle&lt;'scope, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for AtomicCell&lt;T&gt;","synthetic":false,"types":[]},{"text":"impl&lt;T:&nbsp;?Sized&gt; UnwindSafe for ShardedLock&lt;T&gt;","synthetic":false,"types":[]}];
implementors["num_traits"] = [{"text":"impl UnwindSafe for ParseFloatError","synthetic":true,"types":[]},{"text":"impl UnwindSafe for FloatErrorKind","synthetic":true,"types":[]}];
implementors["rayon"] = [{"text":"impl&lt;T&gt; UnwindSafe for IntoIter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; UnwindSafe for Iter&lt;'a, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: RefUnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; !UnwindSafe for Drain&lt;'a, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;K, V&gt; UnwindSafe for IntoIter&lt;K, V&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;K: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;V: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, K, V&gt; UnwindSafe for Iter&lt;'a, K, V&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;K: RefUnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;V: RefUnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, K, V&gt; !UnwindSafe for IterMut&lt;'a, K, V&gt;","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for IntoIter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; UnwindSafe for Iter&lt;'a, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: RefUnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;K, V&gt; UnwindSafe for IntoIter&lt;K, V&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;K: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;V: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, K, V&gt; UnwindSafe for Iter&lt;'a, K, V&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;K: RefUnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;V: RefUnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, K, V&gt; !UnwindSafe for IterMut&lt;'a, K, V&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'a, K, V&gt; !UnwindSafe for Drain&lt;'a, K, V&gt;","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for IntoIter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; UnwindSafe for Iter&lt;'a, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: RefUnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; !UnwindSafe for Drain&lt;'a, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for IntoIter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; UnwindSafe for Iter&lt;'a, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: RefUnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; !UnwindSafe for IterMut&lt;'a, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for IntoIter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; UnwindSafe for Iter&lt;'a, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: RefUnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; !UnwindSafe for IterMut&lt;'a, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; !UnwindSafe for Drain&lt;'a, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;A, B&gt; UnwindSafe for Chain&lt;A, B&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;A: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;B: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; UnwindSafe for Chunks&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; UnwindSafe for Cloned&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; UnwindSafe for Copied&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for Empty&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; UnwindSafe for Enumerate&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, P&gt; UnwindSafe for Filter&lt;I, P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, P&gt; UnwindSafe for FilterMap&lt;I, P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, F&gt; UnwindSafe for FlatMap&lt;I, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, F&gt; UnwindSafe for FlatMapIter&lt;I, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; UnwindSafe for Flatten&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; UnwindSafe for FlattenIter&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, ID, F&gt; UnwindSafe for Fold&lt;I, ID, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;ID: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, U, F&gt; UnwindSafe for FoldWith&lt;I, U, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;U: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, F&gt; UnwindSafe for Inspect&lt;I, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, J&gt; UnwindSafe for Interleave&lt;I, J&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;J: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, J&gt; UnwindSafe for InterleaveShortest&lt;I, J&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;J: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; UnwindSafe for Intersperse&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;&lt;I as ParallelIterator&gt;::Item: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; UnwindSafe for MaxLen&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; UnwindSafe for MinLen&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, F&gt; UnwindSafe for Map&lt;I, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, INIT, F&gt; UnwindSafe for MapInit&lt;I, INIT, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;INIT: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, T, F&gt; UnwindSafe for MapWith&lt;I, T, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for MultiZip&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for Once&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; UnwindSafe for PanicFuse&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;Iter&gt; UnwindSafe for IterBridge&lt;Iter&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;Iter: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, P&gt; UnwindSafe for Positions&lt;I, P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;P: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for Repeat&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for RepeatN&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; UnwindSafe for Rev&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; UnwindSafe for Skip&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;D, S&gt; UnwindSafe for Split&lt;D, S&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;D: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;S: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; UnwindSafe for Take&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, U, ID, F&gt; UnwindSafe for TryFold&lt;I, U, ID, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;ID: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;U: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, U, F&gt; UnwindSafe for TryFoldWith&lt;I, U, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;&lt;U as Try&gt;::Ok: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I, F&gt; UnwindSafe for Update&lt;I, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; UnwindSafe for WhileSome&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;A, B&gt; UnwindSafe for Zip&lt;A, B&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;A: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;B: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;A, B&gt; UnwindSafe for ZipEq&lt;A, B&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;A: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;B: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;I&gt; UnwindSafe for StepBy&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for IntoIter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; UnwindSafe for Iter&lt;'a, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: RefUnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; !UnwindSafe for IterMut&lt;'a, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for Iter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for Iter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for IntoIter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; UnwindSafe for Iter&lt;'a, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: RefUnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a, T&gt; !UnwindSafe for IterMut&lt;'a, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'data, T&gt; UnwindSafe for Iter&lt;'data, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: RefUnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'data, T&gt; UnwindSafe for Chunks&lt;'data, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: RefUnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'data, T&gt; UnwindSafe for ChunksExact&lt;'data, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: RefUnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'data, T&gt; UnwindSafe for Windows&lt;'data, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: RefUnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'data, T&gt; !UnwindSafe for IterMut&lt;'data, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'data, T&gt; !UnwindSafe for ChunksMut&lt;'data, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'data, T&gt; !UnwindSafe for ChunksExactMut&lt;'data, T&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'data, T, P&gt; UnwindSafe for Split&lt;'data, T, P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;P: UnwindSafe,<br>&nbsp;&nbsp;&nbsp;&nbsp;T: RefUnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'data, T, P&gt; !UnwindSafe for SplitMut&lt;'data, T, P&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'ch&gt; UnwindSafe for Chars&lt;'ch&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'ch&gt; UnwindSafe for CharIndices&lt;'ch&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'ch&gt; UnwindSafe for Bytes&lt;'ch&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'ch&gt; UnwindSafe for EncodeUtf16&lt;'ch&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'ch, P&gt; UnwindSafe for Split&lt;'ch, P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;P: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'ch, P&gt; UnwindSafe for SplitTerminator&lt;'ch, P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;P: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'ch&gt; UnwindSafe for Lines&lt;'ch&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'ch&gt; UnwindSafe for SplitWhitespace&lt;'ch&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'ch, P&gt; UnwindSafe for Matches&lt;'ch, P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;P: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'ch, P&gt; UnwindSafe for MatchIndices&lt;'ch, P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;P: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'a&gt; !UnwindSafe for Drain&lt;'a&gt;","synthetic":true,"types":[]},{"text":"impl&lt;T&gt; UnwindSafe for IntoIter&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: UnwindSafe,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;'data, T&gt; !UnwindSafe for Drain&lt;'data, T&gt;","synthetic":true,"types":[]}];
implementors["rayon_core"] = [{"text":"impl !UnwindSafe for ThreadBuilder","synthetic":true,"types":[]},{"text":"impl&lt;'scope&gt; !UnwindSafe for Scope&lt;'scope&gt;","synthetic":true,"types":[]},{"text":"impl&lt;'scope&gt; !UnwindSafe for ScopeFifo&lt;'scope&gt;","synthetic":true,"types":[]},{"text":"impl !UnwindSafe for ThreadPool","synthetic":true,"types":[]},{"text":"impl !UnwindSafe for ThreadPoolBuildError","synthetic":true,"types":[]},{"text":"impl&lt;S&nbsp;=&nbsp;DefaultSpawn&gt; !UnwindSafe for ThreadPoolBuilder&lt;S&gt;","synthetic":true,"types":[]},{"text":"impl !UnwindSafe for Configuration","synthetic":true,"types":[]},{"text":"impl UnwindSafe for FnContext","synthetic":true,"types":[]}];
if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()