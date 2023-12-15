import sys
import folium
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView

class KMLViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visor de KML")
        self.setGeometry(100, 100, 800, 600)

        # Crear un mapa de Folium
        self.mapa = folium.Map(location=[0, 0], zoom_start=1)

        # Crear un objeto QWebEngineView para mostrar el mapa
        self.web_view = QWebEngineView()
        self.web_view.setHtml(self.mapa._repr_html_())

        # Crear un botón para cargar el KML
        self.cargar_kml_button = QPushButton("Cargar KML")
        self.cargar_kml_button.clicked.connect(self.cargar_kml)

        # Crear un diseño vertical para organizar los elementos
        layout = QVBoxLayout()
        layout.addWidget(self.web_view)
        layout.addWidget(self.cargar_kml_button)

        # Crear un widget contenedor y configurar el diseño
        container = QWidget()
        container.setLayout(layout)

        # Configurar el widget contenedor como el widget central de la ventana
        self.setCentralWidget(container)

    def cargar_kml(self):
        # Reemplaza con el nombre de tu archivo KML
        kml_file = 'puntos.kml'

        # Cargar el archivo KML y obtener las coordenadas
        coordenadas = [(p.geometry.coordinates[1], p.geometry.coordinates[0]) for p in folium.Kml(kml_file).features()]

        # Borrar el mapa actual
        self.mapa = folium.Map(location=[0, 0], zoom_start=1)

        # Agregar marcadores para las coordenadas del KML
        for lat, lon in coordenadas:
            folium.Marker([lat, lon], popup='Punto').add_to(self.mapa)

        # Actualizar el contenido del QWebEngineView
        self.web_view.setHtml(self.mapa._repr_html_())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KMLViewer()
    window.show()
    sys.exit(app.exec_())
