# Chapter 2

**1. If we want to use each thread in a grid to calculate one output element of a vector addition, what would be the expression for mapping the thread/block indices to the data index (i)?**
   (A) i=threadIdx.x + threadIdx.y;
   (B) i=blockIdx.x + threadIdx.x;
   (C) i=blockIdx.x * blockDim.x + threadIdx.x;
   (D) i=blockIdx.x * threadIdx.x;

* **Thinking Process:**
    * In CUDA, we have three levels of hierarchy: Grid, Block, and Thread.
    * `threadIdx.x` is the index of the thread within its thread block (usually starting from 0).
    * `blockIdx.x` is the index of the thread block within the grid (usually starting from 0).
    * `blockDim.x` is the number of threads in a thread block.
    * To assign a unique and contiguous data index `i` to each thread in the grid, we need to consider the block it belongs to and its position within that block.
    * For block `blockIdx.x`, all the preceding blocks (from 0 to `blockIdx.x - 1`) collectively contain `blockIdx.x * blockDim.x` threads.
    * Therefore, the first thread in block `blockIdx.x` should process the element at index `blockIdx.x * blockDim.x`.
    * The thread `threadIdx.x` within the block then processes the `threadIdx.x`-th element within that block (counting from 0).
    * Thus, the global data index `i` processed by thread `threadIdx.x` in block `blockIdx.x` is `blockIdx.x * blockDim.x + threadIdx.x`.

* **Answer and Guidance:**
    * The correct answer is **(C)**.
    * **(Guidance):** This is the most basic and crucial indexing method in CUDA, which we call the "Global Thread Index". Imagine thread blocks as rows of boxes, with each box containing `blockDim.x` small balls (threads). To give each ball a unique number `i`, you first count how many full boxes come before the current one (`blockIdx.x` boxes), and since each box has `blockDim.x` balls, there are a total of `blockIdx.x * blockDim.x` balls before it. Then, add the current ball's number within its own box (`threadIdx.x`), and you get the globally unique number `i`.
    * Option (A) only applies to calculations within a 2D thread block and doesn't consider the block index.
    * Option (B) doesn't multiply by the block size, leading to threads in different blocks calculating the same `i` value.
    * The multiplication logic in option (D) is incorrect and cannot guarantee the uniqueness and contiguity of the indices.
---

**2. Assume that we want to use each thread to calculate two adjacent elements of a vector addition. What would be the expression for mapping the thread/block indices to the data index (i) of the first element to be processed by a thread?**
   (A) i=blockIdx.x * blockDim.x + threadIdx.x + 2;
   (B) i=blockIdx.x * threadIdx.x * 2;
   (C) i=(blockIdx.x * blockDim.x + threadIdx.x) * 2;
   (D) i=blockIdx.x * blockDim.x * 2 + threadIdx.x;

* **Thought Process:**
    * First, we still need to calculate the unique global ID for each thread, just like in the previous question: `globalThreadId = blockIdx.x * blockDim.x + threadIdx.x`.
    * Now, each thread needs to handle *two* adjacent elements.
    * Thread 0 handles elements 0 and 1, thread 1 handles elements 2 and 3, thread 2 handles elements 4 and 5, and so on.
    * It can be seen that the index of the first element processed by the `globalThreadId`-th thread is `globalThreadId * 2`.
    * Substituting the expression for `globalThreadId`, we get the index of the first element as `i = (blockIdx.x * blockDim.x + threadIdx.x) * 2`.

* **Answer and Guidance:**
    * The correct answer is **(C)**.
    * **(Guidance):** This question is an advancement based on the first question. When a thread needs to process multiple elements, we usually first calculate the unique global ID of this thread (which is the answer to the first question), and then determine the range of data it is responsible for based on the number of elements processed by each thread (here it is 2). Since each thread processes 2 elements, the index of the first element processed by the $k$-th thread (with a global ID of $k$) is $k * 2$. Replacing $k$ with the familiar `blockIdx.x * blockDim.x + threadIdx.x`, we get answer (C).
    * Option (A) simply adds 2 to the global ID, which does not conform to the mapping relationship required by the question.
    * Option (B) uses an incorrect way to calculate the global ID.
    * The multiplication position in option (D) is incorrect. It is equivalent to multiplying the starting position of the block by 2, but the intra-block offset is not multiplied by 2, which would lead to discontinuous indices.

---

**3. We want to use each thread to calculate two elements of a vector addition. Each thread block processes 2 * blockDim.x consecutive elements that form two sections. All threads in each block will process a section first, each processing one element. They will then all move to the next section, each processing one element. Assume that variable i should be the index for the first element to be processed by a thread. What would be the expression for mapping the thread/block indices to data index of the first element?**
   (A) i=blockIdx.x * blockDim.x + threadIdx.x + 2;
   (B) i=blockIdx.x * threadIdx.x * 2;
   (C) i=(blockIdx.x * blockDim.x + threadIdx.x) * 2;
   (D) i=blockIdx.x * blockDim.x * 2 + threadIdx.x;

* **Thinking Process:**
    * Understand the problem: Each block is responsible for `2 * blockDim.x` consecutive elements. The work is divided into two steps:
        1.  **Step 1 (Section 1):** `blockDim.x` threads within the block, each thread processes one element. Thread `t` (0 <= t < `blockDim.x`) processes the $t^{th}$ element within the range responsible by this block.
        2.  **Step 2 (Section 2):** `blockDim.x` threads within the block, each thread processes one element. Thread `t` processes the $(t + blockDim.x)^{th}$ element within the range responsible by this block.
    * The question asks for the index `i` of the *first* element processed by a thread.
    * We only need to focus on the mapping in the first step (Section 1).
    * Block `blockIdx.x` is responsible for the elements starting from `blockIdx.x * (2 * blockDim.x)`.
    * In the first step, thread `threadIdx.x` processes the `threadIdx.x`-th element within this block's range (relative index).
    * Therefore, the absolute index `i` of this element = (starting index of the block) + (relative index of the thread within the block) = `blockIdx.x * (2 * blockDim.x) + threadIdx.x`.

* **Answer and Guidance:**
    * The correct answer is **(D)**.
    * **(Guidance):** This question describes a slightly more complex memory access pattern, which we can call "Strided Access" or segmented processing. The key is to understand the data range handled by each block and which element is accessed by a thread in the first stage.
    * Each block processes `2 * blockDim.x` elements, so the data processed by the `blockIdx.x`-th block starts from index `blockIdx.x * 2 * blockDim.x`.
    * In the first stage, all threads in parallel process the first half of the block's responsible range (the first "section"). Thread `threadIdx.x` is responsible for processing the `threadIdx.x`-th element starting from the beginning of this block's range.
    * Therefore, the global index `i` is the starting index of the block `blockIdx.x * 2 * blockDim.x` plus the thread's offset within the block `threadIdx.x`.
    * Option (C) is the answer to the previous question, which applies to the case where each thread consecutively processes two elements, which is different from the segmented processing pattern described in this question.

---

**4. For a vector addition, assume that the vector length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel call to have a minimum number of thread blocks to cover all output elements. How many threads will be in the grid?**
   (A) 8000
   (B) 8196
   (C) 8192
   (D) 8200

* **Thinking Process:**
    * We need enough threads to process 8000 elements.
    * Each thread processes 1 element.
    * Each thread block contains 1024 threads.
    * Calculate the number of thread blocks needed: `NumberOfBlocks = ceil(TotalElements / ThreadsPerBlock)`
    * `NumberOfBlocks = ceil(8000 / 1024)`
    * `8000 / 1024 = 7.8125`
    * `ceil(7.8125) = 8`. We need to launch 8 thread blocks to cover all 8000 elements.
    * Total number of threads in the grid = Number of blocks * Threads per block
    * `TotalThreadsInGrid = NumberOfBlocks * ThreadsPerBlock = 8 * 1024 = 8192`.

* **Answer and Guidance:**
    * The correct answer is **(C)**.
    * **(Guidance):** This question tests how to configure the grid based on the task size and block size. Since the thread block size is fixed (1024) and our task size (8000) is not an integer multiple of the block size, we must launch enough blocks to ensure the last element is also processed. Calculating `8000 / 1024` yields a value greater than 7, meaning 7 blocks are insufficient, and 8 blocks must be launched. Even if only a portion of the threads in the 8th block (the first `8000 - 7 * 1024 = 8000 - 7168 = 832` threads) will actually perform valid calculations (usually checked using `if(i < N)`), we still need to launch the full 8 blocks. Therefore, the total number of threads actually launched in the grid is `8 * 1024 = 8192`.
    * Option (A) is the number of elements to be processed, not the total number of threads launched.
    * The other options are incorrect calculation results.

---

**5. If we want to allocate an array of v integer elements in the CUDA device global memory, what would be an appropriate expression for the second argument of the cudaMalloc call?**
   (A) n
   (B) v
   (C) n * sizeof(int)
   (D) v * sizeof(int)

* **Thought Process:**
    * Review the function prototype of `cudaMalloc`: `cudaError_t cudaMalloc(void** devPtr, size_t size);`
    * The first parameter is a pointer to a device pointer.
    * The second parameter `size` is the amount of memory to allocate, in **bytes**.
    * We want to allocate `v` integers.
    * Each integer occupies `sizeof(int)` bytes.
    * Therefore, the total number of bytes needed is `v * sizeof(int)`.

* **Answer and Guidance:**
    * The correct answer is **(D)**.
    * **(Guidance):** The `cudaMalloc` function needs to know how many *bytes* of memory you want to allocate. The question states that we want to allocate `v` elements of type `int`. In C/C++, `sizeof(int)` gives you the number of bytes occupied by an `int` type variable. Therefore, the total number of bytes needed to allocate `v` integers is `v` multiplied by `sizeof(int)`. Always remember that the second parameter of `cudaMalloc` is the number of bytes, not the number of elements!
    * Options (A) and (C) use the incorrect variable name `n`.
    * Option (B) only passes the number of elements `v` and does not multiply it by the size of each element, so the unit is incorrect.

---

**6. If we want to allocate an array of n floating-point elements and have a floating-point pointer variable A_d to point to the allocated memory, what would be an appropriate expression for the first argument of the cudaMalloc () call?**
   (A) n
   (B) (void*) A_d
   (C) A_d
   (D) (void**)&A_d

* **Thinking Process:**
    * Look at the prototype of `cudaMalloc` again: `cudaError_t cudaMalloc(void** devPtr, size_t size);`
    * The type of the first parameter `devPtr` is `void**`, which is a pointer to a `void` pointer.
    * `cudaMalloc` needs to know where to store the address of the allocated device memory. It modifies the pointer variable provided by the caller through this `void**` parameter.
    * We have a `float* A_d` variable, which is used to store the address of the device memory.
    * We need to pass the *address* of the `A_d` variable to `cudaMalloc`, so that `cudaMalloc` can modify the value of `A_d` and make it point to the newly allocated memory.
    * The address of `A_d` is `&A_d`. Since `A_d` is of type `float*`, the type of `&A_d` is `float**`.
    * The first parameter of `cudaMalloc` requires the type `void**`, so we need to cast the `float**` type of `&A_d` to `void**`. That is `(void**)&A_d`.

* **Answer and Guidance:**
    * The correct answer is **(D)**.
    * **(Guidance):** This is another key point of `cudaMalloc`. The role of `cudaMalloc` is to allocate memory and return the address of the allocated memory to you. It does not return the address through the function's return value, but by modifying the pointer variable you pass to it. Therefore, you cannot directly pass `A_d` (which is the current value of `A_d`, possibly NULL or uninitialized), but you must pass the memory address of the `A_d` variable itself, which is `&A_d`. Because the interface definition of `cudaMalloc`'s first parameter is `void**`, we need to perform a type cast `(void**)&A_d`. In this way, `cudaMalloc` internally can find your `A_d` variable through this address and write the address of the allocated device memory into it.
    * Option (A) is the number of elements, which is irrelevant to the first parameter.
    * Options (B) and (C) pass the value of `A_d`. `cudaMalloc` cannot modify the `A_d` in the caller's scope through a value.

---

**7. If we want to copy 3000 bytes of data from host array A_h (A_h is a pointer to element 0 of the source array) to device array A_d (A_d is a pointer to element 0 of the destination array), what would be an appropriate API call for this data copy in CUDA?**
   (A) cudaMemcpy(3000, A_h, A_d, cudaMemcpyHostToDevice);
   (B) cudaMemcpy(A_h, A_d, 3000, cudaMemcpyDeviceToHost);
   (C) cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);
   (D) cudaMemcpy(3000, A_d, A_h, cudaMemcpyHostToDevice);

* **Thought Process:**
    * Review the `cudaMemcpy` function prototype: `cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);`
    * The parameter order is: `destination pointer (dst)`, `source pointer (src)`, `number of bytes to copy (count)`, `copy direction (kind)`.
    * The question requires copying from host `A_h` to device `A_d`.
        * The destination is `A_d`.
        * The source is `A_h`.
        * The number of bytes to copy (count) is `3000`.
        * The copy direction (kind) is from host to device, which is `cudaMemcpyHostToDevice`.
    * Putting these parameters into `cudaMemcpy` in order: `cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);`

* **Answer and Guidance:**
    * The correct answer is **(C)**.
    * **(Guidance):** `cudaMemcpy` is a core CUDA function used to transfer data between host memory and device memory, or between different device memory locations. Remembering its parameter order is crucial: **destination, source, size, direction**. It's similar to how we usually think about copying files, first stating where to copy to (destination), and then stating where to copy from (source). Here, we want to copy data from the host (`A_h`) to the device (`A_d`), so `A_d` is the destination, `A_h` is the source, the size is `3000` bytes, and the direction is `cudaMemcpyHostToDevice`. This corresponds to option (C).
    * Options (A) and (D) have the incorrect parameter order, placing the size before the source or destination.
    * Option (B) has the incorrect copy direction; `cudaMemcpyDeviceToHost` is for copying from the device back to the host.

---

**8. How would one declare a variable err that can appropriately receive the returned value of a CUDA API call?**
   (A) int err;
   (B) cudaError err;
   (C) cudaError_t err;
   (D) cudaSuccess_t err;

* **Thinking Process:**
    * Almost all CUDA Runtime API calls (such as `cudaMalloc`, `cudaMemcpy`, `cudaFree`, and `cudaDeviceSynchronize` or `cudaGetLastError` after kernel launch) return an error code.
    * Consulting the CUDA documentation or based on experience, the standard type for this error code is `cudaError_t`.
    * `cudaSuccess` is an enumeration value of the `cudaError_t` type, indicating successful operation.

* **Answer and Guidance:**
    * The correct answer is **(C)**.
    * **(Guidance):** CUDA API functions typically return a status code to indicate whether the operation was executed successfully. The official data type for this status code is `cudaError_t`. This is an enumeration type defined in the CUDA header files, containing many possible error codes, such as `cudaSuccess` (indicating success), `cudaErrorMemoryAllocation` (indicating memory allocation failure), and so on. Therefore, to receive this return value and check it, we should declare a variable of type `cudaError_t`.
    * Option (A) `int`: Although `cudaError_t` might be implemented using an integer at the underlying level, using `int` would lose type information and is not the best practice.
    * Option (B) `cudaError`: This is part of the type name but lacks the `_t` suffix, so it is not the complete type name.
    * Option (D) `cudaSuccess_t`: `cudaSuccess` is a *value* of the `cudaError_t` type, representing a successful status, not a type itself.

---

**9. Consider the following CUDA kernel and the corresponding host function that calls it: ... (code provided)**
   **a. What is the number of threads per block?**
   **b. What is the number of threads in the grid?**
   **c. What is the number of blocks in the grid?**
   **d. What is the number of threads that execute the code on line 02?**
   **e. What is the number of threads that execute the code on line 04?**

* **Code Analysis:**
    * Kernel launch syntax: `kernel_name<<< gridDim, blockDim, sharedMemBytes, stream >>>(...);`
    * Launch from Host function: `foo_kernel<<<(N + 128 - 1)/128, 128>>>(a_d, b_d, N);`
    * Where `N = 200000`.

* **Solutions:**
    * **a. Number of threads per block (`blockDim`):** This is the second parameter in `<<<...>>>`, which is **128**.
    * **c. Number of blocks in the grid (`gridDim`):** This is the first parameter in `<<<...>>>`, which is `(N + 128 - 1) / 128`.
        * Substituting `N = 200000`: `(200000 + 128 - 1) / 128 = 200127 / 128`.
        * This is integer division, and the result is `floor(200127 / 128.0) = floor(1563.49...) = 1563`.
        * Therefore, there are **1563** blocks in the grid. (The notation `(N + B - 1) / B` is a common integer arithmetic trick to calculate `ceil(N / B)`).
    * **b. Number of threads in the grid:** Total threads = Number of blocks * Number of threads per block = `gridDim * blockDim`.
        * `1563 * 128 = 200064`.
        * Therefore, a total of **200064** threads are launched in the grid.
    * **d. Number of threads that execute line 02:** `unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;`
        * This line of code calculates the global thread index, and every launched thread needs to execute it to determine its identity.
        * Therefore, all threads launched in the grid will execute this line of code. The number is **200064**.
    * **e. Number of threads that execute line 04:** `b[i] = 2.7f * a[i] - 4.3f;`
        * This line of code is inside the `if(i < N)` conditional statement.
        * Only when the global index `i` calculated by a thread is less than `N` (200000) will that thread execute this calculation.
        * The calculated range of `i` is from 0 to `200064 - 1`.
        * The threads for which `i < 200000` will have `i` values from 0 to 199999.
        * Therefore, a total of **200000** threads will execute this line of code (i.e., the threads whose indices correspond to valid data elements).

* **Answers and Guidance:**
    * **a. Threads per block:** **128** (directly read from the second parameter of the kernel launch `< ... , 128 >`)
    * **b. Threads in the grid:** **200064** (calculated as `gridDim * blockDim = 1563 * 128`)
    * **c. Blocks in the grid:** **1563** (calculated from the first parameter of the kernel launch `(200000 + 128 - 1) / 128`)
    * **d. Threads executing line 02:** **200064** (all launched threads execute the index calculation)
    * **e. Threads executing line 04:** **200000** (only threads with index `i` within the valid data range `0` to `N-1` execute the core computation)
    * **(Guidance):** This question comprehensively tests the understanding of kernel launch configurations and basic kernel code analysis.
        * `<<<gridDim, blockDim>>>` is key, where `gridDim` is the number of blocks and `blockDim` is the number of threads per block.
        * The common `(N + blockDim - 1) / blockDim` notation for calculating `gridDim` is used to achieve ceiling division, ensuring enough blocks to cover `N` elements.
        * The total number of threads is `gridDim * blockDim`.
        * Not all code inside a kernel is executed by every thread. Initialization code like index calculation is usually executed by all threads, while core computations or memory access operations are often protected by boundary checks like `if (i < N)` to ensure only threads processing valid data execute them, preventing out-of-bounds access.

---

**10. A new summer intern was frustrated with CUDA. He has been complaining that CUDA is very tedious. He had to declare many functions that he plans to execute on both the host and the device twice, once as a host function and once as a device function. What is your response?**

* **Thinking Process:**
    * The intern's problem is code duplication: the same logic for a function needs to be written in two versions, one for the host (CPU) and one for the device (GPU).
    * CUDA C++ provides a mechanism to solve this problem.
    * Recalling CUDA's function execution space specifiers:
        * `__global__`: Defines a kernel function, called from the host and executed on the device.
        * `__device__`: Defines a device function, can only be called from the device (`__global__` or other `__device__` functions), and executed on the device.
        * `__host__`: Defines a host function, can only be called from the host and executed on the host (this is the default behavior for C/C++ functions and can usually be omitted).
    * The key is that `__host__` and `__device__` can be used in combination.

* **Answer and Guidance:**
    * **(Response):** "I understand how you feel; it can indeed be tedious if you really need to write the same functionality twice. However, CUDA C++ has considered this situation! You can use the combination of `__host__ __device__` keywords to decorate a function. When you do this, the compiler will automatically generate two versions of the code for this function: one version that can run on the host (CPU) and another version that can run on the device (GPU). This way, you only need to write and maintain one source code."

    * **(Additional Explanation, Optional):** "For example, if you have a simple helper function, like calculating the sum of squares of two numbers:
        ```c++
        __host__ __device__ float sum_of_squares(float x, float y) {
            return x*x + y*y;
        }
        ```
        With this definition of the `sum_of_squares` function, you can call it both in your regular host C++ code and in your `__global__` kernel functions or other `__device__` functions, which is very convenient. This avoids the problem of code duplication."

    * **(Summary):** "So, next time you find yourself needing a function that can be used on both the host and the device, remember to add the `__host__ __device__` specifiers to it. This can greatly simplify your code."
