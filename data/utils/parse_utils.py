def parse_range(input_str):
    # Split the input string by commas
    elements = input_str.split(',')
    result = []

    for element in elements:
        if '-' in element:
            # It's a range; split it into start and end
            start, end = element.split('-')
            # Convert start and end to integers and create the range
            result.extend(range(int(start), int(end) + 1))
        else:
            # It's a single integer, just append it
            result.append(int(element))

    return result