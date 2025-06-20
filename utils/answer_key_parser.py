def parse_answer_key(file_path):
    answer_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('.')
            if len(parts) == 2:
                q_no = int(parts[0])
                ans = parts[1].strip().upper()
                answer_dict[q_no] = ans
    return answer_dict
