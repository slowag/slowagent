def calculate_sum(filename):
    try:
        total = 0
        with open(filename, 'r') as file:
            for line in file:
                number = float(line.split()[0])
                total+= number
            return total
    except FileNotFoundError:
        print(f"File {filename} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
nums1 = ['output_par{}.txt'.format(i) for i in range(2,10,10)]
nums2 = ['output_seq{}.txt'.format(i) for i in range(2,10,10)]

for num1, num2 in zip(nums1, nums2):
    av1 = calculate_sum(num1)
    av2 = calculate_sum(num2)
    print((av2 - av1)*100/av1)

