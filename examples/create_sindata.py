import argparse
import numpy as np
import argparse

   

def sine_data_generation(no, seq_len, t_max, main_frequency):
    """Sine data generation.

    Args:
        - no: the number of samples
        - seq_len: sequence length of the time-series
        - dim: feature dimensions

    Returns:
        - data: generated data
    """

    t_step = t_max / seq_len
    time = np.arange(0, t_max, t_step)
    # Initialize the output
    data = np.zeros(shape=(no, seq_len + 2), dtype=np.float32)

    # Generate sine data
    for i in range(no):
        # For each feature

        # Randomly drawn frequency and phase
        freq = main_frequency + np.random.uniform(-0.01, 0.01) # normal variation of 0.02 Hz around main_frequency 
        data[i, -1] = freq
        phase = np.random.uniform(-3, 3) # phase variation around 0 of 3 degres
        data[i, -2] = phase
        
        # Generate sine signal based on the drawn frequency and phase
        data[i, :-2] = np.sin(2*np.pi*freq*time+phase*np.pi/180)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse two optional float arguments: number of samples and sequence length."
    )

    parser.add_argument('outname', help='filename to store csv file')

  
    parser.add_argument("-n",
        "--num_samples", type=int, default=1000, help="Number of samples."
    )

    
   
    parser.add_argument(
        "-s", "--seq_len",
        type=int,
        default=600,
        help="duration of the sequence in s",
    )

   
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=60,
        help="duration of the sequence in s",
    )

    parser.add_argument(
        "-f", "--freq",
        type=float,
        default=0.05,
        help="main frequency in Hz (0.05)",
    )

    args = parser.parse_args()

    print(f"Number of samples: {args.num_samples}")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Duration: {args.duration}")
    print(f"Frequency: {args.freq}")

    data = sine_data_generation(args.num_samples, args.seq_len, args.duration, args.freq)
    np.savetxt(args.outname, data, delimiter=',')