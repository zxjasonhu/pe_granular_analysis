import pandas as pd


def find_positive_sequences(df):
    # Filter the DataFrame for pe_present_on_image equals 1
    df_positive = df[df["pe_present_on_image"] == 1]

    # List to store the results
    sequences = []

    # Group by 'SeriesInstanceUID'
    grouped = df_positive.groupby("SeriesInstanceUID")

    for name, group in grouped:
        # Sort the group by slice_index to ensure continuity
        group = group.sort_values("slice_index")

        # Initialize variables to track the sequences
        start_index = None
        last_index = None
        current_sequence_length = 0

        # Iterate over the group
        for row in group.itertuples():
            # Check if the current index is the next index in the sequence
            if last_index is None or row.slice_index == last_index + 1:
                # Update the last index
                last_index = row.slice_index

                # Update the sequence length
                current_sequence_length += 1

                # Check if the start index is None
                if start_index is None:
                    start_index = row.slice_index
            else:
                # If the next index is not the next in the sequence, reset the sequence
                sequences.append(
                    {
                        "SeriesInstanceUID": name,
                        "StartIndex": start_index,
                        "EndIndex": last_index,
                        "Length": current_sequence_length,
                    }
                )
                start_index = row.slice_index
                last_index = row.slice_index
                current_sequence_length = 1

        # Add the last sequence
        sequences.append(
            {
                "SeriesInstanceUID": name,
                "StartIndex": start_index,
                "EndIndex": last_index,
                "Length": current_sequence_length,
            }
        )

    return pd.DataFrame(sequences)
