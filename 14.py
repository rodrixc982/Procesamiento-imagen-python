import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
from matplotlib import pyplot as plt


class LarvaCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Conteo de Larvas IIAP")
        self.imagen = None
        self.mostrar_gris = False

        self.frame = tk.Frame(self.root, bg='white')
        self.frame.pack(padx=20, pady=20)

        self.label_imagen = tk.Label(self.frame)
        self.label_imagen.pack()

        self.boton_agregar = tk.Button(self.frame, text="Agregar imagen", command=self.agregar_imagen, bg='lightblue', fg='black')
        self.boton_agregar.pack(pady=10)

        self.boton_escala_gris = tk.Button(self.frame, text="Mostrar en escala de grises", command=self.toggle_escala_gris, bg='lightblue', fg='black')
        self.boton_escala_gris.pack(pady=5)

        self.boton_contar = tk.Button(self.frame, text="Contar larvas", command=self.contar_larvas, bg='lightgreen', fg='black')
        self.boton_contar.pack(pady=10)

        self.boton_histograma_rgb = tk.Button(self.frame, text="Histograma RGB", command=self.mostrar_histograma_rgb, bg='lightyellow', fg='black')
        self.boton_histograma_rgb.pack(pady=5)

        self.boton_histograma_cmyk = tk.Button(self.frame, text="Histograma CMYK", command=self.mostrar_histograma_cmyk, bg='lightyellow', fg='black')
        self.boton_histograma_cmyk.pack(pady=5)

        self.boton_histograma_lab = tk.Button(self.frame, text="Histograma LAB", command=self.mostrar_histograma_lab, bg='lightyellow', fg='black')
        self.boton_histograma_lab.pack(pady=5)

        self.boton_histograma_mapa_bits = tk.Button(self.frame, text="Histograma Mapa de bits", command=self.mostrar_histograma_mapa_bits, bg='lightyellow', fg='black')
        self.boton_histograma_mapa_bits.pack(pady=5)

        self.boton_duotono = tk.Button(self.frame, text="Mostrar en modo Duotono", command=self.mostrar_duotono, bg='lightpink', fg='black')
        self.boton_duotono.pack(pady=5)

        self.boton_color_indexado = tk.Button(self.frame, text="Mostrar en modo Color indexado", command=self.mostrar_color_indexado, bg='lightpink', fg='black')
        self.boton_color_indexado.pack(pady=5)

        self.boton_multicanal = tk.Button(self.frame, text="Mostrar en modo Multicanal", command=self.mostrar_multicanal, bg='lightpink', fg='black')
        self.boton_multicanal.pack(pady=5)

    def agregar_imagen(self):
        ruta_imagen = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=(("Archivos de imagen", "*.jpg;*.jpeg;*.png"), ("Todos los archivos", "*.*"))
        )

        if ruta_imagen:
            self.imagen = cv2.imread(ruta_imagen)
            self.mostrar_imagen(self.imagen)

    def toggle_escala_gris(self):
        self.mostrar_gris = not self.mostrar_gris

        if self.mostrar_gris:
            imagen_gris = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)
            self.mostrar_imagen(imagen_gris)
            self.boton_escala_gris.config(text="Mostrar en color")
        else:
            self.mostrar_imagen(self.imagen)
            self.boton_escala_gris.config(text="Mostrar en escala de grises")

    def mostrar_imagen(self, imagen):
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        imagen_resized = cv2.resize(imagen_rgb, (400, 300))
        imagen_pil = Image.fromarray(imagen_resized)
        imagen_tk = ImageTk.PhotoImage(imagen_pil)

        self.label_imagen.configure(image=imagen_tk)
        self.label_imagen.image = imagen_tk

    def contar_larvas(self):
        if self.imagen is None:
            messagebox.showerror("Error", "Por favor, seleccione una imagen.")
            return

        gray = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_larvas = len(contours)

        img_with_contours = self.imagen.copy()
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)
        cv2.putText(img_with_contours, f"Total de larvas: {total_larvas}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        self.mostrar_imagen(img_with_contours)
        messagebox.showinfo("Conteo de Larvas", f"Total de larvas: {total_larvas}")

    def mostrar_histograma_rgb(self):
        if self.imagen is None:
            messagebox.showerror("Error", "Por favor, seleccione una imagen.")
            return

        pixels_rgb = self.imagen.reshape(-1, 3)
        r, g, b = cv2.split(self.imagen)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(r.flatten(), bins=256, color='r', alpha=0.5, label='R')
        ax.hist(g.flatten(), bins=256, color='g', alpha=0.5, label='G')
        ax.hist(b.flatten(), bins=256, color='b', alpha=0.5, label='B')
        ax.set_xlabel('Valor')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Histograma RGB')
        ax.legend()

        plt.show()

    def mostrar_histograma_cmyk(self):
        if self.imagen is None:
            messagebox.showerror("Error", "Por favor, seleccione una imagen.")
            return

        imagen_cmyk = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2CMYK)
        c, m, y, k = cv2.split(imagen_cmyk)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(c.flatten(), bins=256, color='c', alpha=0.5, label='C')
        ax.hist(m.flatten(), bins=256, color='m', alpha=0.5, label='M')
        ax.hist(y.flatten(), bins=256, color='y', alpha=0.5, label='Y')
        ax.hist(k.flatten(), bins=256, color='k', alpha=0.5, label='K')
        ax.set_xlabel('Valor')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Histograma CMYK')
        ax.legend()

        plt.show()

    def mostrar_histograma_lab(self):
        if self.imagen is None:
            messagebox.showerror("Error", "Por favor, seleccione una imagen.")
            return

        imagen_lab = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(imagen_lab)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(l.flatten(), bins=256, color='lightgray', alpha=0.5, label='L')
        ax.hist(a.flatten(), bins=256, color='green', alpha=0.5, label='A')
        ax.hist(b.flatten(), bins=256, color='blue', alpha=0.5, label='B')
        ax.set_xlabel('Valor')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Histograma LAB')
        ax.legend()

        plt.show()

    def mostrar_histograma_mapa_bits(self):
        if self.imagen is None:
            messagebox.showerror("Error", "Por favor, seleccione una imagen.")
            return

        imagen_bw = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)
        _, imagen_binaria = cv2.threshold(imagen_bw, 127, 255, cv2.THRESH_BINARY)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(imagen_binaria.flatten(), bins=2, color='black', alpha=0.5)
        ax.set_xlabel('Valor')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Histograma Mapa de bits')

        plt.show()

    def mostrar_duotono(self):
        if self.imagen is None:
            messagebox.showerror("Error", "Por favor, seleccione una imagen.")
            return

        imagen_gris = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)
        _, imagen_duotono = cv2.threshold(imagen_gris, 127, 255, cv2.THRESH_BINARY)

        self.mostrar_imagen(imagen_duotono)

    def mostrar_color_indexado(self):
        if self.imagen is None:
            messagebox.showerror("Error", "Por favor, seleccione una imagen.")
            return

        imagen_indexada = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2RGB)
        imagen_indexada = cv2.resize(imagen_indexada, (400, 300), interpolation=cv2.INTER_NEAREST)

        self.mostrar_imagen(imagen_indexada)

    def mostrar_multicanal(self):
        if self.imagen is None:
            messagebox.showerror("Error", "Por favor, seleccione una imagen.")
            return

        b, g, r = cv2.split(self.imagen)

        zeros = np.zeros(b.shape, dtype=np.uint8)
        blue_channel = cv2.merge((zeros, zeros, r))
        green_channel = cv2.merge((zeros, g, zeros))
        red_channel = cv2.merge((b, zeros, zeros))

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(blue_channel)
        axs[0].set_title('Canal Azul')
        axs[0].axis('off')
        axs[1].imshow(green_channel)
        axs[1].set_title('Canal Verde')
        axs[1].axis('off')
        axs[2].imshow(red_channel)
        axs[2].set_title('Canal Rojo')
        axs[2].axis('off')

        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = LarvaCounterApp(root)
    root.mainloop()








