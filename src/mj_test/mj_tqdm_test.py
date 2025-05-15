from tqdm import tqdm
filtered_data = [1,2,3]
input_list = ['a','b','c']
output_list = ['x','y','z']

for item, input_prompt, result in tqdm(zip(filtered_data, input_list, output_list)):
    print(item, input_prompt, result)

