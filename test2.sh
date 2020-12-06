#!/bin/bash -e
echo "Compiling"
gcc vm.c -o vm
echo "Running vm FIFO"
./vm BACKING_STORE.bin addresses.txt > fifo.txt
echo "Running vm LRU"
./vm BACKING_STORE.bin addresses.txt -l lru > lru.txt
echo "Comparing"
diff fifo.txt lru.txt

