    def get_free_gpu(self):
        """
        Identifies the GPU with the most free memory using 'nvidia-smi' and returns its index.

        This function queries the available GPUs on the system and determines which one has 
        the highest amount of free memory. It uses the `nvidia-smi` command-line tool to gather 
        GPU memory usage data. If successful, it returns the index of the GPU with the most free memory.
        If the query fails or an error occurs, it returns None.

        Returns:
        int: Index of the GPU with the most free memory, or None if no GPU is found or an error occurs.
        """
        try:
            # Run nvidia-smi to query GPU memory usage and free memory
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
            gpu_info = result.stdout.decode('utf-8').strip().split('\n')

            free_gpu = None
            max_free_memory = 0
            for i, info in enumerate(gpu_info):
                used, free = map(int, info.split(','))
                if free > max_free_memory:
                    max_free_memory = free
                    free_gpu = i
            return free_gpu
        except Exception as e:
            print(f"Error finding free GPU: {e}")
            return None
