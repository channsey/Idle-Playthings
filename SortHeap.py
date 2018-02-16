import time
import numpy as np
from collections import deque
import queue

def HeapSort(A):
    def heapify(A):
        i = (len(A)-2) // 2
        while i >= 0:
            siftDown(A,i,len(A))
            i -= 1
    def siftDown(A, root, end):
        while root * 2 + 1 < end:
            node = root * 2 + 1
            #check big exists & if big > small
            if node + 1 < end and A[node] < A[node + 1]:
                node += 1 #become big
            #check if big or small is bigger than root
            if node <= end and A[root] < A[node]:
                A[root], A[node] = A[node], A[root]
                root = node
            else:
                return
    def getRuns(A, B, root, end):
        print('%.2f' % A[root] )
        B.append(A[root])
        if root * 2 + 2 < end:
            if A[root * 2 + 1] <  A[root * 2 + 2]:
                getRuns(A,B, root * 2 + 2, end)
                getRuns(A,B, root * 2 + 1, end)
                print("runr")
            else:
                getRuns(A,B, root * 2 + 1, end)
                getRuns(A,B, root * 2 + 2, end)
                print("runl")
        elif root * 2 + 1 < end:
            #print("rune") 
            getRuns(A,B, root * 2 + 1, end)
            print("rune") 
        #print('%.2f' % A[root] )
    def getRunners(A, B, C, root, end):
        print(C)#print('%.2f' % A[root])
        B.append(A[root])
        if root * 2 + 2 < end:
            if A[root * 2 + 1] <  A[root * 2 + 2]:
                getRunners(A,B,C, root * 2 + 2, end)
                getRunners(A,B,C, root * 2 + 1, end)
                print("runr")
                C = merge(C, B)
            else:
                getRunners(A,B,C, root * 2 + 1, end)
                getRunners(A,B,C, root * 2 + 2, end)
                print("runl")
                C = deque(merge(C, B))
        elif root * 2 + 1 < end:
            getRunners(A,B,C, root * 2 + 1, end)
            print("rune")
            C = merge(C, B) 

    def merge(big, small):
        result = deque([])
        if len(big) == 0:
            big = small
            return small
        while 0 < len(small):
            if small[0] < big[0]:
                result.append(big.popleft())#print(big[j])
            else:
                result.append(small.popleft())#print(small[i])
        while 0 < len(big):
            result.append(big.popleft())#print(big[j])
        return result   
                
    
    big = deque([ 12, 10, 8, 6, 4, 2, 0])      
    small = deque([ 9, 7, 5, 3, 1])
    big = merge(big, small)
    print(big)
    """
    print("A =[", end='')
    for i in A:
        print(' %.2f' % i, end='')
    print("]")
    start = time.process_time()
    """
    heapify(A)
    """
    print(time.process_time() - start)
    start = time.process_time()
    heapify2(B)
    print(time.process_time() - start)
    start = time.process_time()
    heapify3(C)
    print(time.process_time() - start)
    start = time.process_time()
    heapify4(F)
    print(time.process_time() - start)
    """
    B = deque([])
    C = deque([])
    getRunners(A,B,C,0, len(A))
    print("B =[", end='')
    for i in B:
        print(' %.2f' % i, end='')
    print("]")
    """
    end = len(A) - 1
    while end > 0:
        A[end], A[0] = A[0], A[end]
        end -= 1
        siftDown1(A, 0, end)
    """


if __name__ == '__main__':
    #T = np.random.random_sample((16,))
    T = [13, 14, 94, 33, 82, 25, 59, 94, 65, 23, 45, 27, 73, 25, 39, 10] 
    HeapSort(T)