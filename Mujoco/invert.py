import numpy as np
import copy


# Invert a binary tree represented as a list (BFS)
def invert(arr):
    h = int(np.log2(len(arr) + 1))
    for d in range(h):
        for i in range(d):

            # Indexes to swap
            idx_1 = 2 ** d - 1 + i
            idx_2 = idx_1 + 2 ** (d - i) - 1
            # Swap
            temp = arr[idx_1]
            arr[idx_1] = arr[idx_2]
            arr[idx_2] = temp

            #size_move = 2 ** (d - i) - 1
            #temp = arr[2 ** d - 1 + i]
            #arr[2 ** d - 1 + i] = arr[2**d-1+i + size_move]
            #arr[2 ** d - 1 + i + size_move] = temp

    print(arr)
    return arr


invert([2, 1, 3])
invert([4, 2, 7, 1, 3, 6, 9])


# Array with elements in {0, 1, None}.
# E.g.: [0, 1, None, None]
# Instead of replacing None with 0 or 1, do both!
# And then return all possible arrays.
# So for the above example, return [[0,1,0,0], [0,1,1,0], [0,1,1,1], [0,1,0,1]]
def extend_arr(arr):

    arr = [arr]
    for i, val in enumerate(arr[0]):

        if val is None:
            arr.extend(copy.deepcopy(arr))

            for j in range(len(arr)):
                if j < len(arr) / 2:
                    arr[j][i] = 0
                else:
                    arr[j][i] = 1
    print(arr)
    return arr



class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i, v1 in enumerate(nums):
            for j, v2 in enumerate(nums[i:]):
                if v1 + v2 == target:
                    return [i, j]

extend_arr([1, 1, None, None])
