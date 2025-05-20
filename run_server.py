from tcp_server import TCPServer
import signal
import sys
import time

def signal_handler(sig, frame):
    """
    Handle interrupt signals (Ctrl+C) for graceful shutdown.
    
    This function:
    1. Prints a shutdown message
    2. Stops the server
    3. Exits the program
    
    Args:
        sig: Signal number
        frame: Current stack frame
    """
    print("\nStopping server...")
    server.stop()
    sys.exit(0)

if __name__ == "__main__":
    """
    Main entry point for the TCP server.
    
    This script:
    1. Creates a TCP server instance
    2. Sets up signal handling for graceful shutdown
    3. Starts the server
    4. Keeps the main thread alive until interrupted
    """
    # Create and start TCP server with default parameters:
    # - Port: 5005
    # - Window size: 200 samples (2 seconds at 100Hz)
    # - Sampling rate: 100Hz
    server = TCPServer(port=5005, window_size=200, sampling_rate=100)
    
    # Set up signal handler for graceful shutdown
    # This allows the server to clean up resources when Ctrl+C is pressed
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start the server
        server.start()
        print("Press Ctrl+C to stop the server")
        
        # Keep the main thread alive
        # On Windows, we use a simple loop with sleep
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Handle keyboard interrupt explicitly
        server.stop() 