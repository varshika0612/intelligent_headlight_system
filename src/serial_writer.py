import serial
import time
import numpy as np

class MatrixSerialWriter:
    def __init__(self, port='COM3', baudrate=115200):
        """
        Initializes the serial connection to the ESP32.
        Change 'COM3' to your actual port (e.g., '/dev/ttyUSB0' on Linux/Mac).
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.connect()

    def connect(self):
        try:
            # timeout=0.05 ensures non-blocking behavior in your main CV loop
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.05)
            time.sleep(2)  # Critical: wait for ESP32 to reboot after serial connection opens
            print(f"[SerialWriter] Successfully connected to ESP32 on {self.port}")
        except serial.SerialException as e:
            print(f"[SerialWriter] WARNING: Could not connect to {self.port}. Running in simulation mode. ({e})")
            self.serial_conn = None

    def send_mask(self, mask_8x8: np.ndarray):
        """
        Packs the 8x8 boolean mask into bytes and sends it to the hardware.
        """
        if self.serial_conn is None:
            return  # Fail gracefully if hardware isn't plugged in

        # According to your README: True = LED OFF (glare zone), False = LED ON.
        # We invert this so 1 = ON, 0 = OFF for the hardware logic.
        led_states = ~np.array(mask_8x8, dtype=bool)

        # Pack 8 boolean values into 1 uint8 byte per row.
        # This converts a 64-bool array into just 8 bytes for ultra-fast serial transfer.
        packed_bytes = np.packbits(led_states, axis=1).flatten()

        # Frame format: [START_MARKER] [8 DATA BYTES] [END_MARKER]
        # This prevents the ESP32 from getting misaligned if a byte drops
        frame = bytearray([0xAA]) + bytearray(packed_bytes) + bytearray([0x55])

        try:
            self.serial_conn.write(frame)
        except serial.SerialTimeoutException:
            print("[SerialWriter] Write timeout.")
        except Exception as e:
            print(f"[SerialWriter] Write error: {e}")

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("[SerialWriter] Connection closed.")

# --- Quick Test Block ---
if __name__ == "__main__":
    writer = MatrixSerialWriter(port='COM3') # Update port as needed
    
    # Create a test mask (True = Glare/OFF, False = Road/ON)
    test_mask = np.zeros((8, 8), dtype=bool)
    test_mask[2:5, 3:6] = True # Simulate a car in the middle
    
    print("Sending test mask...")
    print(test_mask)
    writer.send_mask(test_mask)
    
    time.sleep(1)
    writer.close()