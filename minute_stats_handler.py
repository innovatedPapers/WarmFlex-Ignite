"""
Minute Stats Handler Module
Handles processing and writing of per-minute container statistics
"""
import csv
import os

# Global variables for minute tracking
current_minute = None

def process_list(my_list):
    """
    Group data by minute and calculate element-wise averages for each minute
    
    Args:
        my_list: List of sublists where first element is the minute
        
    Returns:
        List of averaged sublists, one per minute
    """
    if not my_list:
        return []
    
    # Group by minute (first element)
    minute_groups = {}
    for sublist in my_list:
        minute = sublist[0]
        if minute not in minute_groups:
            minute_groups[minute] = []
        minute_groups[minute].append(sublist)
    
    # Calculate average for each minute
    result = []
    for minute in sorted(minute_groups.keys()):
        sublists = minute_groups[minute]
        
        # Calculate element-wise average for this minute
        sublist_length = len(sublists[0])
        averages = []
        
        for i in range(sublist_length):
            position_sum = sum(sublist[i] for sublist in sublists)
            average = position_sum / len(sublists)
            averages.append(average)
        
        result.append(averages)
    
    return result

def process_and_write_minute_stats(data_to_process, output_dir=".", method_name="warmFlex"):
    """
    Process minute stats data and write to CSV file
    
    Args:
        data_to_process: List of minute stats data to process
        output_dir: Directory to write output files
        method_name: Name of the method to include in filename (default: "warmFlex")
    """
    
    if not data_to_process:
        return
        
    # Process the data (get averages)
    result = process_list(data_to_process)
    
    # Write to output file with dynamic method name
    min_file = os.path.join(output_dir, f"result_containers_{method_name}.csv")
    with open(min_file, 'a', newline='') as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["Minute","Available Containers",
                        "Busy Containers","Existance Containers"])
        if result:
            for stat in result:
                w.writerow(stat)


def check_minute_change(chunk_min_time, per_minute_stats, output_dir="result", method_name="warmFlex"):
    """
    Check if minute changed and handle completed minutes
    
    Args:
        chunk_min_time: Minimum arrival time for current chunk
        per_minute_stats: Global list of per-minute statistics
        output_dir: Directory to write output files
        method_name: Name of the method to include in filename (default: "warmFlex")
        
    Returns:
        Updated per_minute_stats list with completed minutes removed
    """
    
    global current_minute
    
    tick_minute = int(chunk_min_time)

    # Initialize cursor so the very first call can flush minute (tick_minute - 1) if present
    if current_minute is None:
        current_minute = tick_minute - 1

    # If time hasn't advanced, nothing to do
    if tick_minute <= current_minute:
        return per_minute_stats

    # Flush all completed minutes strictly before tick_minute
    completed_data = [row for row in per_minute_stats if row[0] < tick_minute]
    remaining_data = [row for row in per_minute_stats if row[0] >= tick_minute]

    if completed_data:
        process_and_write_minute_stats(completed_data, output_dir, method_name)

    # Move cursor forward
    current_minute = tick_minute
    return remaining_data


def finalize_minute_stats(per_minute_stats, output_dir="result", method_name="warmFlex"):
    """
    Process and write any remaining minute stats at the end of simulation
    
    Args:
        per_minute_stats: Global list of per-minute statistics
        output_dir: Directory to write output files
        method_name: Name of the method to include in filename (default: "warmFlex")
        
    Returns:
        Empty list (all data processed)
    """
    if per_minute_stats:
        process_and_write_minute_stats(per_minute_stats, output_dir, method_name)
    return []

