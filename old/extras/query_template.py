import re

object_to_label_mapping = {
    "PERSON": 0,
    "BICYCLE": 1,
    "CAR": 2,
    "MOTORCYCLE": 3,
    "AIRPLANE": 4,
    "BUS": 5,
    "TRAIN": 6,
    "TRUCK": 7,
    "BOAT": 8,
    "TRAFFIC LIGHT": 9,
    "FIRE HYDRANT": 10,
    "STREET SIGN": 11,
    "STOP SIGN": 12,
    "PARKING METER": 13
}

def parse_query(query_string):
    # Define regular expressions for different query patterns
    select_object_pattern = r"SELECT (\w+) FROM (\w+).(\w+) BEGIN (\d+) TO (\d+) IN (\d+)"
    select_color_object_pattern = r"SELECT (\w+) (\w+) FROM (\w+).(\w+) BEGIN (\d+) TO (\d+) IN (\d+)"
    select_overtaking_object_pattern = r"SELECT OVERTAKING (\w+) FROM (\w+).(\w+) BEGIN (\d+) TO (\d+) IN (\d+)"
    select_overtaking_c_object_pattern = r"SELECT OVERTAKING (\w+) (\w+) FROM (\w+).(\w+) BEGIN (\d+) TO (\d+) IN (\d+)"

    # Try to match each pattern
    match = re.match(select_object_pattern, query_string)
    if match:
        return {
            "query_type": "select_object",
            "object_name": match.group(1),
            "video_name": match.group(2),
            "video_extension": match.group(3),
            "starting_time": int(match.group(4)),
            "ending_time": int(match.group(5)),
            "window_size": int(match.group(6)),
            "class_id" : int(object_to_label_mapping[match.group(1)])
        }

    match = re.match(select_color_object_pattern, query_string)
    if match:
        if(match.group(1)!="OVERTAKING"):
            return {
                "query_type": "select_color_object",
                "color_name": match.group(1),
                "object_name": match.group(2),
                "video_name": match.group(3),
                "video_extension": match.group(4),
                "starting_time": int(match.group(5)),
                "ending_time": int(match.group(6)),
                "window_size": int(match.group(7)),
                "class_id" : int(object_to_label_mapping[match.group(2)])
            }

    match = re.match(select_overtaking_object_pattern, query_string)
    if match:
        return {
            "query_type": "select_overtaking_object",
            "object_name": match.group(1),
            "video_name": match.group(2),
            "video_extension": match.group(3),
            "starting_time": int(match.group(4)),
            "ending_time": int(match.group(5)),
            "window_size": int(match.group(6)),
            "class_id" : int(object_to_label_mapping[match.group(1)])
        }
    
    match = re.match(select_overtaking_c_object_pattern, query_string)
    if match:
        return {
            "query_type": "select_overtaking_c_object",
            "color_name": match.group(1),
            "object_name": match.group(2),
            "video_name": match.group(3),
            "video_extension": match.group(4),
            "starting_time": int(match.group(5)),
            "ending_time": int(match.group(6)),
            "window_size": int(match.group(7)),
            "class_id" : int(object_to_label_mapping[match.group(2)])
        }

    return None  # No matching pattern

# Test the function with example query strings
query1 = "SELECT CAR FROM input.mp4 BEGIN 0 TO 30 IN 5"
query2 = "SELECT RED TRUCK FROM input.mp4 BEGIN 0 TO 30 IN 5"
query3 = "SELECT OVERTAKING MOTORCYCLE FROM input.mp4 BEGIN 0 TO 30 IN 5"
query4 = "SELECT OVERTAKING BLUE BUS FROM input.mp4 BEGIN 0 TO 30 IN 5"

print(parse_query(query1))
print(parse_query(query2))
print(parse_query(query3))
print(parse_query(query4))
