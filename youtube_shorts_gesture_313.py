import cv2
import pyautogui
import time
import numpy as np
import sys
from enum import Enum

class GestureType(Enum):
    NONE = 0
    SWIPE_UP = 1
    SWIPE_DOWN = 2
    PALM_OPEN = 3
    FIST = 4

class HandDetector:
    def __init__(self):
        # Parameter untuk deteksi tangan
        self.skin_lower = np.array([0, 20, 70], dtype=np.uint8)
        self.skin_upper = np.array([20, 255, 255], dtype=np.uint8)
        self.min_contour_area = 5000
        self.motion_threshold = 3000
        
    def detect_hand(self, frame):
        """Mendeteksi keberadaan tangan menggunakan deteksi warna kulit"""
        # Convert ke HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Buat mask untuk warna kulit
        skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        
        # Operasi morfologi untuk membersihkan noise
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        
        # Temukan kontur
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, skin_mask
        
        # Ambil kontur terbesar
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Jika area terlalu kecil, abaikan
        if cv2.contourArea(largest_contour) < self.min_contour_area:
            return None, None, skin_mask
        
        # Dapatkan bounding box dan centroid
        x, y, w, h = cv2.boundingRect(largest_contour)
        centroid_x = x + w // 2
        centroid_y = y + h // 2
        
        return (centroid_x, centroid_y), largest_contour, skin_mask
    
    def detect_gesture(self, frame, prev_hand_pos, current_hand_pos):
        """Mendeteksi gesture berdasarkan pergerakan tangan"""
        if current_hand_pos is None:
            return GestureType.NONE, None
        
        cx, cy = current_hand_pos
        
        # Deteksi bentuk tangan (palm vs fist)
        hand_roi = frame[max(0, cy-50):min(frame.shape[0], cy+50), 
                         max(0, cx-50):min(frame.shape[1], cx+50)]
        
        if hand_roi.size == 0:
            return GestureType.NONE, (cx, cy)
        
        # Deteksi palm open berdasarkan area dan circularity
        is_palm = self._is_palm_open(hand_roi)
        
        if is_palm:
            return GestureType.PALM_OPEN, (cx, cy)
        
        # Deteksi swipe gesture
        if prev_hand_pos:
            prev_x, prev_y = prev_hand_pos
            delta_y = prev_y - cy
            
            if delta_y > 50:  # Swipe up
                return GestureType.SWIPE_UP, (cx, cy)
            elif delta_y < -50:  # Swipe down
                return GestureType.SWIPE_DOWN, (cx, cy)
        
        return GestureType.NONE, (cx, cy)
    
    def _is_palm_open(self, hand_roi):
        """Mendeteksi apakah telapak tangan terbuka"""
        if hand_roi.size == 0:
            return False
        
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return False
        
        # Circularity calculation
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Palm typically has lower circularity than fist
        return circularity < 0.5 and area > 500

class YouTubeShortsGestureControl:
    def __init__(self):
        # Inisialisasi webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Tidak dapat mengakses webcam")
            sys.exit(1)
        
        # Inisialisasi detector tangan
        self.detector = HandDetector()
        
        # Variabel untuk melacak gesture
        self.prev_hand_pos = None
        self.last_action_time = 0
        self.action_cooldown = 1.5  # Cooldown 1.5 detik antara aksi
        self.gesture_cooldown = 0
        
        # Frame history untuk motion detection
        self.prev_frame = None
        
        print("YouTube Shorts Gesture Control - Python 3.13.6")
        print("Pastikan browser dengan YouTube Shorts sedang aktif")
        print("Gesture yang didukung:")
        print("- Gerakan tangan ke atas: Next Shorts")
        print("- Gerakan tangan ke bawah: Previous Shorts")
        print("- Telapak tangan terbuka: Play/Pause")
        print("- Tekan 'q' untuk keluar")

    def perform_action(self, gesture_type):
        """Melakukan aksi berdasarkan gesture yang terdeteksi"""
        current_time = time.time()
        if current_time - self.last_action_time < self.action_cooldown:
            return
        
        if gesture_type == GestureType.SWIPE_UP:
            print("Gesture: Swipe up - Next Shorts")
            pyautogui.press('down')  # Scroll ke bawah untuk next shorts
            self.last_action_time = current_time
            
        elif gesture_type == GestureType.SWIPE_DOWN:
            print("Gesture: Swipe down - Previous Shorts")
            pyautogui.press('up')  # Scroll ke atas untuk previous shorts
            self.last_action_time = current_time
            
        elif gesture_type == GestureType.PALM_OPEN:
            print("Gesture: Palm open - Play/Pause")
            pyautogui.press('space')  # Tekan spasi untuk play/pause
            self.last_action_time = current_time

    def draw_ui(self, frame, gesture_type, hand_pos):
        """Menggambar UI pada frame"""
        h, w, _ = frame.shape
        
        # Gambar informasi gesture
        if gesture_type != GestureType.NONE and hand_pos:
            cx, cy = hand_pos
            color = (0, 255, 0)  # Hijau untuk gesture terdeteksi
            
            if gesture_type == GestureType.SWIPE_UP:
                cv2.putText(frame, "SWIPE UP", (cx - 50, cy - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.arrowedLine(frame, (cx, cy + 20), (cx, cy - 20), color, 3)
                
            elif gesture_type == GestureType.SWIPE_DOWN:
                cv2.putText(frame, "SWIPE DOWN", (cx - 60, cy - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.arrowedLine(frame, (cx, cy - 20), (cx, cy + 20), color, 3)
                
            elif gesture_type == GestureType.PALM_OPEN:
                cv2.putText(frame, "PALM OPEN", (cx - 50, cy - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.circle(frame, (cx, cy), 25, color, 2)
        
        # Tampilkan instruksi
        instructions = [
            "Gerakan tangan ke atas: Next Shorts",
            "Gerakan tangan ke bawah: Previous Shorts", 
            "Telapak tangan terbuka: Play/Pause",
            "Tekan 'q' untuk keluar"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = h - 80 + i * 25
            cv2.putText(frame, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run(self):
        """Menjalankan loop utama aplikasi"""
        try:
            print("Memulai deteksi gesture...")
            print("Arahkan tangan ke kamera dan coba gesture yang berbeda")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Tidak dapat membaca frame dari webcam")
                    break
                
                # Flip frame secara horizontal untuk mirror effect
                frame = cv2.flip(frame, 1)
                
                # Deteksi tangan
                hand_pos, contour, mask = self.detector.detect_hand(frame)
                
                # Deteksi gesture
                gesture_type, current_hand_pos = self.detector.detect_gesture(
                    frame, self.prev_hand_pos, hand_pos
                )
                
                # Lakukan aksi jika gesture terdeteksi
                if gesture_type != GestureType.NONE and self.gesture_cooldown <= 0:
                    self.perform_action(gesture_type)
                    self.gesture_cooldown = 20  # Cooldown untuk mencegah deteksi berulang
                else:
                    self.gesture_cooldown -= 1
                
                # Update posisi tangan sebelumnya
                self.prev_hand_pos = hand_pos
                
                # Gambar kontur tangan jika terdeteksi
                if contour is not None:
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                    if hand_pos:
                        cx, cy = hand_pos
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
                # Gambar UI
                self.draw_ui(frame, gesture_type, hand_pos)
                
                # Tampilkan frame
                cv2.imshow('YouTube Shorts Gesture Control', frame)
                
                # Tampilkan mask (untuk debugging)
                if hand_pos:
                    cv2.imshow('Hand Mask', mask)
                
                # Keluar dengan menekan 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            print("Aplikasi dihentikan")

if __name__ == "__main__":
    # Setel keamanan PyAutoGUI
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1
    
    gesture_control = YouTubeShortsGestureControl()
    gesture_control.run()