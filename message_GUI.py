
def MessageBox(text = "", value = None):
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(text, value)
    root.destroy()


def ButtonMessageBox(text = "", value = None):
    import tkinter as tk
    from tkinter import messagebox
    def show_message():
        messagebox.showinfo(text, value)
    root = tk.Tk()
    root.title("Example App")
    button = tk.Button(root, text="Show Message", command=show_message)
    button.pack(pady=20)
    root.mainloop()
