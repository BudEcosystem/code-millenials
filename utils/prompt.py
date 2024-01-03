
def get_prompt(prompt, response=''):
    return f"""You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

### Instruction:
{prompt}

### Response:
{response}"""