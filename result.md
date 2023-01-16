# 3 results

Print results:

```bash
cat 3.perf_test.head_200.log | grep 'elapsed_time:\|Namespace'
```

Whole result:

```log
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=1, eval_threads_cnt=1, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 8.931770086288452 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=1, eval_threads_cnt=2, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 12.17116641998291 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=1, eval_threads_cnt=4, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 20.440920114517212 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=1, eval_threads_cnt=8, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 48.69748377799988 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=2, eval_threads_cnt=1, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 9.205172777175903 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=2, eval_threads_cnt=2, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 12.652450561523438 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=2, eval_threads_cnt=4, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 21.631831884384155 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=2, eval_threads_cnt=8, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 38.24245476722717 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=4, eval_threads_cnt=1, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 8.71142053604126 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=4, eval_threads_cnt=2, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 12.939996242523193 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=4, eval_threads_cnt=4, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 22.10081195831299 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=4, eval_threads_cnt=8, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 41.619277477264404 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=8, eval_threads_cnt=1, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 11.274406671524048 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=8, eval_threads_cnt=2, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 19.47328209877014 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=8, eval_threads_cnt=4, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 24.388059616088867 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=8, eval_threads_cnt=8, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 41.363564014434814 seconds

```



### Explaining load results:


```
cat 3.perf_test.head_200.log | grep 'elapsed_time:\|Namespace' | grep 'eval_threads_cnt=1,' -A 1
```

```
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=1, eval_threads_cnt=1, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 8.931770086288452 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=2, eval_threads_cnt=1, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 9.205172777175903 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=4, eval_threads_cnt=1, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 8.71142053604126 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=8, eval_threads_cnt=1, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 11.274406671524048 seconds
```

Best data load was for 4 threads loading data, but it may be problematic to decide precisely because of disk usage of other apps. We can see that the loading process is not getting better with more threads. That may be because of merging graphs, goto code comment 'merge subrgraphs'.




### Explaining eval results:

```
cat 3.perf_test.head_200.log | grep 'elapsed_time:\|Namespace' | grep 'load_threads_cnt=1,' -A 1
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=1, eval_threads_cnt=1, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 8.931770086288452 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=1, eval_threads_cnt=2, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 12.17116641998291 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=1, eval_threads_cnt=4, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 20.440920114517212 seconds
Namespace(file_path='web-BerkStan.head_200.txt', load_threads_cnt=1, eval_threads_cnt=8, run_vis=False, print_nodes=False)
Execution time - elapsed_time: 48.69748377799988 seconds
```

Best evalutaion is for 1 thread, which is dissapointment. I suspect that the problem may be in passing the graph structure to `def node_price(G, v)`


As we can see the execution time is better for 1 load thread and most likely for 1 eval thread.

Better performance may be achieved with reinventing wheel with graph manipulation instead of using `networkx` or extending their work in better way. (Or I may have done something wrong with the parallel implmenentation itself...)

