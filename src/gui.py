import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageGrab
import cv2
import numpy as np
import torch
from torchvision import transforms
import os
import sys

# Add src to path if running from src
sys.path.append(os.path.dirname(__file__))

from model import DigitNet
from preprocess import preprocess_image
from solver import Solver
import threading
import keyboard
import ctypes

class NumberRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nikke AZX Minigame Solver")
        self.root.geometry("1200x800")

        # Top Frame for controls
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.load_btn = tk.Button(self.top_frame, text="Load Image", command=self.load_image)
        self.load_btn.pack(side=tk.LEFT)
        
        self.clipboard_btn = tk.Button(self.top_frame, text="Load from Clipboard", command=self.load_from_clipboard)
        self.clipboard_btn.pack(side=tk.LEFT, padx=10)
        
        self.start_btn = tk.Button(self.top_frame, text="Start Game", command=self.start_game, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(self.top_frame, text="Please use Load Image to load an image.")
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.always_on_top_var = tk.BooleanVar()
        self.always_on_top_check = tk.Checkbutton(self.top_frame, text="Always on Top", 
                                                variable=self.always_on_top_var, command=self.toggle_always_on_top)
        self.always_on_top_check.pack(side=tk.RIGHT, padx=10)
        
        
        self.score_label = tk.Label(self.top_frame, text="")
        self.score_label.pack(side=tk.RIGHT, padx=10)

        # Canvas for image
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Scrollbars removed as per user request (Auto-fit window used)
        # self.v_scroll = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        # self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # self.canvas.configure(yscrollcommand=self.v_scroll.set)
        
        # Data
        self.image_path = None
        self.tk_image = None
        self.original_image_cv = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Game State
        self.matrix = [] # 2D array of {val, item_id, row, col}
        self.num_rows = 0
        self.num_cols = 0
        self.game_active = False
        self.current_solution = None # (r1, c1, r2, c2)
        self.solution_rect_id = None
        self.score = 0
        self.solution_path = []
        self.is_solving = False
        
        # Interaction state
        self.selected_text_id = None
        self.selection_box_id = None
        
        # Load Model
        self.model = DigitNet().to(self.device)
        self.load_model()
        
        # Bindings
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.root.bind("<Key>", self.on_key_press)
        
        # Determine hotkey from settings or default
        try:
            keyboard.add_hotkey('alt+a', lambda: self.root.after(0, self.eliminate_solution))
        except ImportError:
            print("Keyboard library not installed/working properly.")
        except Exception as e:
            print(f"Failed to register hotkey: {e}")

        self.check_admin_privileges()

    def check_admin_privileges(self):
        try:
            is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        except:
            is_admin = False
            
        if not is_admin:
            self.root.title("Nikke AZX Minigame Solver (Non-Admin - Hotkeys may fail in game)")
            # Defer the warning slightly so it doesn't block startup visibly before window appears or strictly after
            self.root.after(500, lambda: messagebox.showwarning(
                "Admin Privileges Recommended", 
                "You are not running as Administrator.\n\n"
                "Global hotkeys (Alt+A) may NOT work when the game window is active "
                "if the game is running as Administrator (common for games).\n\n"
                "Please restart this application as Administrator if hotkeys fail."
            ))

    def load_model(self):
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pth')
        if not os.path.exists(model_path):
             model_path = 'model.pth' # Fallback
             
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print("Model loaded.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
        else:
            messagebox.showwarning("Warning", "model.pth not found. Please train the model first.")

    def load_from_clipboard(self):
        try:
            img = ImageGrab.grabclipboard()
            if isinstance(img, Image.Image):
                # Save to temp file to reuse unique processing logic
                temp_path = "temp_clipboard.png"
                img.save(temp_path)
                
                self.image_path = temp_path
                self.status_label.config(text="Processing clipboard image...")
                self.start_btn.config(state=tk.DISABLED)
                self.game_active = False
                self.score_label.config(text="")
                self.root.update_idletasks()
                
                # Reuse load logic structure
                try:
                    # Load and display original image
                    self.original_image_cv = cv2.imread(self.image_path)
                    if self.original_image_cv is None:
                         # cv2 might fail on some temp formats, reload from PIL
                         # But img.save should work. 
                         # Let's ensure we read it back correctly.
                         self.original_image_cv = cv2.imread(os.path.abspath(self.image_path))
                    
                    cv_rgb = cv2.cvtColor(self.original_image_cv, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(cv_rgb)
                    self.tk_image = ImageTk.PhotoImage(pil_img)
                    
                    # Setup canvas
                    img_w, img_h = pil_img.width, pil_img.height
                    self.canvas.config(scrollregion=(0, 0, img_w, img_h))
                    self.canvas.delete("all") # Clear previous
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
                    
                    # Resize window (Auto-fit)
                    extra_h = 60 
                    extra_w = 40 
                    self.root.geometry(f"{img_w + extra_w}x{img_h + extra_h}")
                    
                    # Process
                    self.process_and_draw(pil_img)
                    self.start_btn.config(state=tk.NORMAL)
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to process clipboard image: {e}")
                    self.status_label.config(text="Error processing image.")

            else:
                messagebox.showwarning("Warning", "No image found in clipboard.")
        except Exception as e:
            messagebox.showerror("Error", f"Clipboard error: {e}")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not file_path:
            return

        self.image_path = file_path
        self.status_label.config(text="Processing...")
        self.start_btn.config(state=tk.DISABLED)
        self.game_active = False
        self.score_label.config(text="")
        self.root.update_idletasks()

        try:
            # Load and display original image
            self.original_image_cv = cv2.imread(self.image_path)
            cv_rgb = cv2.cvtColor(self.original_image_cv, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv_rgb)
            self.tk_image = ImageTk.PhotoImage(pil_img)
            
            # Setup canvas
            img_w, img_h = pil_img.width, pil_img.height
            self.canvas.config(scrollregion=(0, 0, img_w, img_h))
            self.canvas.delete("all") # Clear previous
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            
            # Resize window (Auto-fit)
            extra_h = 60 
            extra_w = 40 
            self.root.geometry(f"{img_w + extra_w}x{img_h + extra_h}")
            
            # Process
            self.process_and_draw(pil_img)
            self.start_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {e}")
            self.status_label.config(text="Error processing image.")

    def process_and_draw(self, pil_image_ref):
        try:
            cells_data, num_rows, num_cols = preprocess_image(self.image_path)
            
            if not cells_data:
                self.status_label.config(text="No digits found.")
                return

            self.status_label.config(text=f"Found {num_rows} rows and ~{num_cols} columns.")
            
            # Initialize Matrix
            # Use max(col) just in case num_cols estimate is off
            max_r = 0
            max_c = 0
            for c in cells_data:
                max_r = max(max_r, c['row'])
                max_c = max(max_c, c['col'])
            
            self.num_rows = max_r + 1
            self.num_cols = max_c + 1
            
            self.matrix = [[None for _ in range(self.num_cols)] for _ in range(self.num_rows)]

            # Prepare batch for recognition
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            batch_tensors = []
            cells_for_pred = [] 
            
            for cell in cells_data:
                cell_img_pil = Image.fromarray(cell['image'])
                tensor = transform(cell_img_pil)
                batch_tensors.append(tensor)
                cells_for_pred.append(cell)

            if batch_tensors:
                batch = torch.stack(batch_tensors).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(batch)
                    _, predictions = torch.max(outputs, 1)
                    predictions = predictions.cpu().numpy()
                
                for i, cell in enumerate(cells_for_pred):
                    pred_digit = int(predictions[i])
                    x, y, w, h = cell['x'], cell['y'], cell['w'], cell['h']
                    r, c = cell['row'], cell['col']
                    
                    text_x = x + w / 2
                    text_y = y + 5
                    
                    item_id = self.canvas.create_text(text_x, text_y, text=str(pred_digit), 
                                            fill="#00008B", font=("Arial", 14, "bold"),
                                            tags=("digit",))
                                            
                    if r < self.num_rows and c < self.num_cols:
                        self.matrix[r][c] = {
                            'val': pred_digit,
                            'item_id': item_id,
                            'rect': (x, y, w, h)
                        }
                                            
        except Exception as e:
            print(e)
            raise e
            
    def start_game(self):
        # Deep copy matrix for solver to ensure thread safety / no mutation of UI state during solve
        # Actually our matrix contains dicts. Solver converts to immutable state anyway.
        # But we should prevent user edits while solving.
        
        self.game_active = False # Will be active after solve
        self.start_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Calculating best solution path... (This might take a moment)")
        self.root.update_idletasks()
        
        self.is_solving = True
        
        # Run in thread
        threading.Thread(target=self.run_solver, daemon=True).start()

    def run_solver(self):
        try:
            solver = Solver()
            
            def update_progress(status_msg):
                self.root.after(0, lambda: self.status_label.config(text=f"Searching... {status_msg}"))
                
            path = solver.solve(self.matrix, progress_callback=update_progress)
            self.root.after(0, self.on_solver_finished, path)
        except Exception as e:
            print(f"Solver error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Solver Error", f"An error occurred: {e}"))
            self.root.after(0, self.on_solver_finished, [])

    def on_solver_finished(self, path):
        self.is_solving = False
        self.solution_path = path
        
        if not path:
             self.status_label.config(text="No solutions found.")
             self.start_btn.config(state=tk.NORMAL)
             return
             
        self.game_active = True
        self.score = 0
        self.score_label.config(text=f"Score: 0")
        
        # Deselect any manual edits
        self.deselect()
        
        self.status_label.config(text=f"Found optimal path with {len(path)} moves. Click to eliminate.")
        self.find_next_solution()

    def find_next_solution(self):
        if not self.solution_path:
            self.current_solution = None
            self.game_active = False
            self.status_label.config(text="No more solutions in path.")
            messagebox.showinfo("Game Over", f"Path Completed! Final Score: {self.score}")
            self.start_btn.config(state=tk.NORMAL)
            return

        # Pop the first move
        self.current_solution = self.solution_path.pop(0)
        self.highlight_solution(self.current_solution)
        self.status_label.config(text=f"Solution found ({len(self.solution_path)} remaining)! Click to eliminate.")

    def highlight_solution(self, solution):
        r1, c1, r2, c2 = solution
        
        min_x, min_y = 99999, 99999
        max_xw, max_yh = 0, 0
        
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                cell = self.matrix[r][c]
                if cell:
                    x, y, w, h = cell['rect']
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_xw = max(max_xw, x + w)
                    max_yh = max(max_yh, y + h)
                
        # Draw rect
        pad = 5
        self.solution_rect_id = self.canvas.create_rectangle(
            min_x - pad, min_y - pad, max_xw + pad, max_yh + pad,
            outline="green", width=3, tags="solution"
        )
        
    def eliminate_solution(self):
        if not self.current_solution:
            return
            
        r1, c1, r2, c2 = self.current_solution
        count = 0
        
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                cell = self.matrix[r][c]
                if cell:
                    self.canvas.delete(cell['item_id'])
                    
                    # Clear pixels (cover with gray rect)
                    x, y, w, h = cell['rect']
                    self.canvas.create_rectangle(x, y, x+w, y+h, fill="gray", outline="gray")
                    
                    self.matrix[r][c] = None # Eliminated
                    count += 1
        
        # Remove highlight
        if self.solution_rect_id:
            self.canvas.delete(self.solution_rect_id)
            self.solution_rect_id = None
            
        self.score += count
        self.score_label.config(text=f"Score: {self.score}")
        self.current_solution = None
        
        # Next
        self.find_next_solution()

    def on_canvas_click(self, event):
        # If Game Active and Solution Waiting
        if self.game_active and self.current_solution:
            self.eliminate_solution()
            return
            
        # Normal Editing Mode
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        curr_item = self.canvas.find_closest(canvas_x, canvas_y)
        if not curr_item:
            return
            
        tags = self.canvas.gettags(curr_item)
        if "digit" in tags:
            self.select_item(curr_item[0])
        else:
            self.deselect()

    def select_item(self, item_id):
        self.deselect()
        self.selected_text_id = item_id
        bbox = self.canvas.bbox(item_id)
        if bbox:
            pad = 2
            self.selection_box_id = self.canvas.create_rectangle(
                bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad,
                outline="red", width=2
            )
            
    def deselect(self):
        if self.selection_box_id:
            self.canvas.delete(self.selection_box_id)
            self.selection_box_id = None
        self.selected_text_id = None

    def on_key_press(self, event):
        if self.selected_text_id and event.char.isdigit():
            new_digit = int(event.char)
            self.canvas.itemconfig(self.selected_text_id, text=str(new_digit))
            
            # Find which cell this text belongs to and update matrix
            found = False
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    cell = self.matrix[r][c]
                    if cell and cell['item_id'] == self.selected_text_id:
                        cell['val'] = new_digit
                        found = True
                        break
                if found: break

    def toggle_always_on_top(self):
        is_on_top = self.always_on_top_var.get()
        self.root.attributes("-topmost", is_on_top)

def main():
    root = tk.Tk()
    app = NumberRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
