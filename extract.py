import json
import sys # Used to print errors to stderr

def format_recidivism_output_jsonl(input_filename, output_filename):
    """
    Reads a JSON Lines file, extracts 'two_year_recid' from each line,
    and writes each result as a separate JSON object on its own line
    to the specified output file (JSON Lines format).

    Args:
        input_filename (str): The path to the input JSON Lines file.
        output_filename (str): The path to the output JSON Lines file.
                                It's conventional to use a .jsonl extension.
    """
    error_lines = []
    processed_count = 0

    try:
        # Open both files using 'with' to ensure they are closed automatically
        # Open output file in write mode ('w') with UTF-8 encoding
        with open(output_filename, 'w', encoding='utf-8') as outfile, \
             open(input_filename, 'r') as infile:

            for i, line in enumerate(infile, 1): # Start line number at 1
                line = line.strip() # Remove leading/trailing whitespace
                if not line:
                    continue # Skip empty lines

                try:
                    # Parse the JSON string from the current line
                    data = json.loads(line)

                    # Extract the value - crucial step
                    recid_value = data['two_year_recid']

                    # Create the new dictionary with only the desired key
                    output_dict = {"two_year_recid": recid_value}

                    # Convert the new dictionary back to a JSON string
                    # ensure_ascii=False is good practice
                    # No indent needed for JSON Lines
                    output_json_string = json.dumps(output_dict, ensure_ascii=False)

                    # Write the JSON string FOLLOWED BY A NEWLINE to the output file
                    outfile.write(output_json_string + '\n')
                    processed_count += 1

                except json.JSONDecodeError:
                    # Print errors to standard error (stderr)
                    print(f"Warning: Could not decode JSON on line {i}. Skipping.", file=sys.stderr)
                    error_lines.append(i)
                except KeyError:
                    # Handle cases where the key might be missing
                    print(f"Warning: Key 'two_year_recid' not found on line {i}. Skipping.", file=sys.stderr)
                    error_lines.append(i)
                except Exception as e: # Catch other potential errors during line processing
                    print(f"Warning: An unexpected error occurred processing line {i}: {e}. Skipping.", file=sys.stderr)
                    error_lines.append(i)

        # If we get here, the files were opened and processed without crashing
        print(f"\nOutput successfully saved to '{output_filename}' (JSON Lines format).", file=sys.stderr)

    except FileNotFoundError:
        # Specifically handle the input file not being found
        print(f"Error: Input file not found at '{input_filename}'", file=sys.stderr)
    except IOError as e:
        # Handle errors related to opening/writing the output file
        print(f"Error: Could not open or write to output file '{output_filename}': {e}", file=sys.stderr)
    except Exception as e:
        # Handle other unexpected errors during file operations
        print(f"An unexpected error occurred during file operations: {e}", file=sys.stderr)

    # --- Optional: Print a summary to stderr after processing is complete ---
    # This summary will run even if there were file errors, but processed_count will reflect reality.
    print(f"\n--- Processing Summary ---", file=sys.stderr)
    print(f"Attempted to process lines from '{input_filename}'.", file=sys.stderr)
    print(f"Successfully processed and wrote {processed_count} records to '{output_filename}'.", file=sys.stderr)
    if error_lines:
        print(f"Encountered errors on {len(error_lines)} lines (see warnings above).", file=sys.stderr)
    elif processed_count > 0:
         print(f"No errors encountered during line processing.", file=sys.stderr)
    elif processed_count == 0 and not error_lines: # Input file might have been empty or unreadable
         print(f"No records processed. Input file might be empty or unreadable, or contained only invalid lines.", file=sys.stderr)


# --- Main execution ---
if __name__ == "__main__":
    # <<< Change these variables to your actual file names/paths >>>
    input_file_path = 'test_data/test.json' # Assuming this is your JSON Lines input

    # <<< Specify the desired name for the output file >>>
    # Using the .jsonl extension is recommended for JSON Lines files
    output_file_path = 'true.json'

    format_recidivism_output_jsonl(input_file_path, output_file_path)

    print(f"\nScript finished. Check '{output_file_path}' for results and console for warnings/errors.", file=sys.stderr)