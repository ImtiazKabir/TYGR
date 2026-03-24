#include <stdlib.h>

// Binary search - similar to training data patterns
int binarySearch(int* arr, int size, int target) {
    int left = 0;
    int right = size - 1;
    int mid;

    while (left <= right) {
        mid = left + (right - left) / 2;
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}

// Sum of array elements
long sumArray(int* nums, int n) {
    long total = 0;
    int i;
    for (i = 0; i < n; i++) {
        total += nums[i];
    }
    return total;
}

// Find maximum element
int findMax(int* arr, int len) {
    int max = arr[0];
    int i;
    for (i = 1; i < len; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

// Count specific values
int countValue(int* data, int size, int val) {
    int count = 0;
    int idx;
    for (idx = 0; idx < size; idx++) {
        if (data[idx] == val) {
            count++;
        }
    }
    return count;
}

// Swap two integers
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}
