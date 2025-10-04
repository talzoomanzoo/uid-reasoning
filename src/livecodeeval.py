def run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, split, data_limit, sample_limit, model_path, thinkseg, step_limit, self_certainty, cot_decoding, confidence, entropy, apply_backoff=False):
    if dataset_name == 'livecode':
        # Prepare samples and generations for codegen_metrics
        samples_list = []
        generations_list = []

        # Collect difficulty levels for per-domain metrics
        difficulties = []
        per_difficulty_count = {}

        #added code for medbullets, qwq-llama-distill
        # output_list = [output_list.choices[i].__dict__.get('text') for i in range(len(output_list))]
        num_valid_answer = 0
        
        for item, input_prompt, result in tqdm(zip(filtered_data, input_list, output_list)):
            difficulty = item.get("difficulty", "Unknown")
            difficulties.append(difficulty)
            
            # Track metrics per domain
            if difficulty not in per_difficulty_count.keys():
                per_difficulty_count[difficulty] = 0

            # Generate sample_limit generations for this sample
            sample_generations = []
            for gen_idx in range(sample_limit):  # Generate sample_limit different generations
                # For each generation, you would typically call the model again
                # For now, using the same result (you'll need to modify this based on your setup)
                item['Output'] = result.outputs[0].text
                pred_code = extract_answer(item['Output'], mode='codegen')
                
                if pred_code != '':
                    num_valid_answer += 1
                    per_difficulty_count[difficulty] += 1
                
                sample_generations.append(pred_code)

            # Assuming each item has 'input_output' with 'inputs' and 'outputs'
            public_test_cases = json.loads(item.get("public_test_cases", "{}"))

            inputs, outputs = [], []
            for case in public_test_cases:
                inputs.append(case["input"])
                outputs.append(case["output"])

            sample = {
                "input_output": json.dumps({
                    "inputs": inputs,
                    "outputs": outputs
                }),
            }

            samples_list.append(sample)
            generations_list.append(sample_generations)  # Add all sample_limit generations
            item['Pred_Answer'] = sample_generations[0]  # Use first generation as main answer
            item['Question'] = input_prompt

            # Calculate proxy metrics for each generation
            for gen_idx in range(sample_limit):
                # Calculate self_certainty, cot_decoding, confidence, entropy for each generation
                if self_certainty:
                    sc_obj = calculate_self_certainty(result.outputs[0].logprobs)
                    item[f'Self_Certainty_{gen_idx}'] = sc_obj
                
                if cot_decoding:
                    cot_decoding_obj = calculate_cot_decoding(result.outputs[0].logprobs)
                    item[f'Cot_Decoding_{gen_idx}'] = cot_decoding_obj
                
                if confidence:
                    confidence_obj = calculate_confidence(result.outputs[0].logprobs)
                    item[f'Confidence_{gen_idx}'] = confidence_obj
                
                if entropy:
                    entropy_obj = calculate_entropy(result.outputs[0].logprobs)
                    item[f'Entropy_{gen_idx}'] = entropy_obj

            # Calculate aggregated proxy metrics
            if self_certainty:
                per_output_self_cert_summary = []
                for gen_idx in range(sample_limit):
                    if f'Self_Certainty_{gen_idx}' in item:
                        # Check if the code execution passed (true) or failed (false)
                        code_passed = bool(item.get('Results', [[]])[0][gen_idx] if item.get('Results') and len(item.get('Results', [[]])[0]) > gen_idx else False)
                        per_output_self_cert_summary.append({
                            f'output_{gen_idx}': item[f'Self_Certainty_{gen_idx}'].get('self_certainty', float('nan')),
                            'math_equal': code_passed,  # Use actual code execution result
                        })
                
                borda_voting_self_cert = calculate_borda_voting_self_certainty(per_output_self_cert_summary)
                item['borda_voting_self_cert'] = borda_voting_self_cert

            if cot_decoding:
                per_output_cot_decoding_summary = []
                for gen_idx in range(sample_limit):
                    if f'Cot_Decoding_{gen_idx}' in item:
                        # Check if the code execution passed (true) or failed (false)
                        code_passed = bool(item.get('Results', [[]])[0][gen_idx] if item.get('Results') and len(item.get('Results', [[]])[0]) > gen_idx else False)
                        per_output_cot_decoding_summary.append({
                            f'output_{gen_idx}': item[f'Cot_Decoding_{gen_idx}'].get('confidence_score', float('nan')),
                            'math_equal': code_passed,  # Use actual code execution result
                        })
                
                highest_cot_decoding = calculate_highest_cot_decoding(per_output_cot_decoding_summary)
                item['highest_cot_decoding'] = highest_cot_decoding

            if confidence:
                per_output_confidence_summary = []
                for gen_idx in range(sample_limit):
                    if f'Confidence_{gen_idx}' in item:
                        # Check if the code execution passed (true) or failed (false)
                        code_passed = bool(item.get('Results', [[]])[0][gen_idx] if item.get('Results') and len(item.get('Results', [[]])[0]) > gen_idx else False)
                        per_output_confidence_summary.append({
                            f'output_{gen_idx}': item[f'Confidence_{gen_idx}'].get('calculate_confidence', float('nan')),
                            'math_equal': code_passed,  # Use actual code execution result
                        })
                
                highest_confidence = calculate_highest_confidence(per_output_confidence_summary)
                item['highest_confidence'] = highest_confidence

            if entropy:
                per_output_entropy_summary = []
                for gen_idx in range(sample_limit):
                    if f'Entropy_{gen_idx}' in item:
                        # Check if the code execution passed (true) or failed (false)
                        code_passed = bool(item.get('Results', [[]])[0][gen_idx] if item.get('Results') and len(item.get('Results', [[]])[0]) > gen_idx else False)
                        per_output_entropy_summary.append({
                            f'output_{gen_idx}': item[f'Entropy_{gen_idx}'].get('calculate_entropy', float('nan')),
                            'math_equal': code_passed,  # Use actual code execution result
                        })
                
                lowest_entropy = calculate_lowest_entropy(per_output_entropy_summary)
                item['lowest_entropy'] = lowest_entropy

        # Call codegen_metrics with pass@k for different k values
        k_list = list(range(1, sample_limit + 1))  # [1, 2, 3, ..., sample_limit]
        metrics, results, final_metadata = codegen_metrics(
            samples_list,
            generations_list,
            k_list=k_list,  # Evaluate pass@1, pass@2, ..., pass@k
            num_process_evaluate=2,   # Parallel evaluation
            timeout=10,  # Set timeout to 10 seconds
            debug=False,  # Enable debug mode
        )

        # Extract pass@k for each k value
        pass_at_k_results = {}
        for k in k_list:
            pass_at_k_results[f'pass@{k}'] = metrics.get(f'pass@{k}', 0.0)

        # For individual results, use pass@1 (the most detailed)
        detail_pass_at_1 = metrics['detail']['pass@1']

        for item, pass1, res, meta in zip(filtered_data, detail_pass_at_1.values(), results.values(), final_metadata):
            item['Metrics'] = {'pass@1': pass1}
            item['Results'] = res
            item['Final_metadata'] = meta

        # Initialize per-difficulty metrics for each k
        difficulty_metrics = defaultdict(lambda: defaultdict(list))
        for idx, difficulty in enumerate(difficulties):
            # For per-difficulty metrics, we'll use pass@1 as the base
            pass1 = detail_pass_at_1[idx]
            difficulty_metrics[difficulty]['pass@1'].append(pass1)

        # Compute overall metrics for each k
        overall_metrics = {}
        for k in k_list:
            overall_metrics[f'pass@{k}'] = pass_at_k_results[f'pass@{k}']

        overall_metrics['num_valid_answer'] = f'{num_valid_answer} of {len(input_list)*sample_limit}'
        overall_metrics['query_latency'] = f'{(total_time / (len(input_list) * sample_limit) * 1000):.0f} s'

        # Compute per-difficulty pass@k for each k
        per_difficulty_metrics = {}
        for difficulty, passes in difficulty_metrics.items():
            # For each difficulty, we need to compute pass@k for each k
            difficulty_pass_at_k = {}
            
            # We need to get the individual results for this difficulty
            difficulty_indices = [i for i, d in enumerate(difficulties) if d == difficulty]
            
            for k in k_list:
                # Extract pass@k for this difficulty
                if f'pass@{k}' in metrics['detail']:
                    detail_pass_at_k = metrics['detail'][f'pass@{k}']
                    difficulty_pass_at_k_values = [detail_pass_at_k[i] for i in difficulty_indices]
                    difficulty_pass_at_k[f'pass@{k}'] = np.mean(difficulty_pass_at_k_values) if len(difficulty_pass_at_k_values) > 0 else 0.0
                else:
                    difficulty_pass_at_k[f'pass@{k}'] = 0.0
            
            num_valid_answer = per_difficulty_count[difficulty]
            per_difficulty_metrics[difficulty] = {
                **difficulty_pass_at_k,  # Include all pass@k values
                'num_valid_answer': f'{num_valid_answer} of {len(difficulty_indices)*sample_limit}'
            }

        # Calculate proxy pass@k using actual metrics
        def calculate_proxy_pass_at_k_with_actual_metrics(filtered_data, difficulties, k_list):
            """
            Calculate pass@k using actual proxy metrics from the evaluation.
            """
            proxy_pass_at_k = {}
            
            for k in k_list:
                proxy_pass_at_k[f'pass@{k}'] = {}
                
                for difficulty in set(difficulties):
                    # Get indices for this difficulty
                    difficulty_indices = [i for i, d in enumerate(difficulties) if d == difficulty]
                    
                    if len(difficulty_indices) == 0:
                        continue
                        
                    # For each question in this difficulty, get the proxy scores and results
                    question_data = {
                        'self_certainty': [],
                        'cot_decoding': [],
                        'confidence': [],
                        'entropy': []
                    }
                    
                    for idx in difficulty_indices:
                        item = filtered_data[idx]
                        
                        # Extract proxy scores and results for this question
                        if 'borda_voting_self_cert' in item and item['borda_voting_self_cert']:
                            # Get the self_certainty score from the selected output
                            selected_output = list(item['borda_voting_self_cert'].keys())[0]
                            if selected_output.startswith('output_'):
                                score = item['borda_voting_self_cert'][selected_output]
                                # Check if the selected output actually passed
                                gen_idx = int(selected_output.split('_')[1])
                                code_passed = bool(item.get('Results', [[]])[0][gen_idx] if item.get('Results') and len(item.get('Results', [[]])[0]) > gen_idx else False)
                                question_data['self_certainty'].append((score, code_passed))
                        
                        if 'highest_cot_decoding' in item and item['highest_cot_decoding']:
                            # Get the cot_decoding score from the selected output
                            selected_output = list(item['highest_cot_decoding'].keys())[0]
                            if selected_output.startswith('output_'):
                                score = item['highest_cot_decoding'][selected_output]
                                # Check if the selected output actually passed
                                gen_idx = int(selected_output.split('_')[1])
                                code_passed = bool(item.get('Results', [[]])[0][gen_idx] if item.get('Results') and len(item.get('Results', [[]])[0]) > gen_idx else False)
                                question_data['cot_decoding'].append((score, code_passed))
                        
                        if 'highest_confidence' in item and item['highest_confidence']:
                            # Get the confidence score from the selected output
                            selected_output = list(item['highest_confidence'].keys())[0]
                            if selected_output.startswith('output_'):
                                score = item['highest_confidence'][selected_output]
                                # Check if the selected output actually passed
                                gen_idx = int(selected_output.split('_')[1])
                                code_passed = bool(item.get('Results', [[]])[0][gen_idx] if item.get('Results') and len(item.get('Results', [[]])[0]) > gen_idx else False)
                                question_data['confidence'].append((score, code_passed))
                        
                        if 'lowest_entropy' in item and item['lowest_entropy']:
                            # Get the entropy score from the selected output
                            selected_output = list(item['lowest_entropy'].keys())[0]
                            if selected_output.startswith('output_'):
                                score = item['lowest_entropy'][selected_output]
                                # Check if the selected output actually passed
                                gen_idx = int(selected_output.split('_')[1])
                                code_passed = bool(item.get('Results', [[]])[0][gen_idx] if item.get('Results') and len(item.get('Results', [[]])[0]) > gen_idx else False)
                                question_data['entropy'].append((score, code_passed))
                    
                    # Calculate pass@k for each proxy metric
                    for proxy_type in ['self_certainty', 'cot_decoding', 'confidence', 'entropy']:
                        if proxy_type not in proxy_pass_at_k:
                            proxy_pass_at_k[proxy_type] = {}
                        
                        if f'pass@{k}' not in proxy_pass_at_k[proxy_type]:
                            proxy_pass_at_k[proxy_type][f'pass@{k}'] = {}
                        
                        if len(question_data[proxy_type]) >= k:
                            # Sort by score (descending for self_certainty, cot_decoding, confidence; ascending for entropy)
                            if proxy_type == 'entropy':
                                sorted_data = sorted(question_data[proxy_type], key=lambda x: x[0])  # ascending for entropy
                            else:
                                sorted_data = sorted(question_data[proxy_type], key=lambda x: x[0], reverse=True)  # descending for others
                            
                            top_k_data = sorted_data[:k]
                            
                            # For pass@k, we need to check if at least one of the top k actually passed
                            proxy_pass_at_k[proxy_type][f'pass@{k}'][difficulty] = 1.0 if any(result for _, result in top_k_data) else 0.0
                        else:
                            proxy_pass_at_k[proxy_type][f'pass@{k}'][difficulty] = 0.0
            
            return proxy_pass_at_k

        # Calculate proxy pass@k using actual metrics
        proxy_metrics = calculate_proxy_pass_at_k_with_actual_metrics(filtered_data, difficulties, k_list)

        # Add proxy metrics to the final metrics
        final_metrics = {
            'overall': overall_metrics,
            'per_domain': per_difficulty_metrics,
            'proxy_metrics': proxy_metrics
        }
        
        # Save metrics to JSON file for LiveCodeBench
        import time
        t = time.localtime()
        result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}-{sample_limit}-thinkseg{thinkseg}-step{step_limit}.json'
        metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}-{sample_limit}-thinkseg{thinkseg}-step{step_limit}.metrics.json'
        
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save the metrics
        with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
            json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

        with open(os.path.join(output_dir, metrics_json_name), mode='w', encoding='utf-8') as json_file:
            json.dump(final_metrics, json_file, indent=4, ensure_ascii=False)
        
        print(f"LiveCodeBench metrics saved to {os.path.join(output_dir, metrics_json_name)}")