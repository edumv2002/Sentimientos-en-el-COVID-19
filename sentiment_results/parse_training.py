import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_training_log(file_path):
    # Patrones para extraer datos
    train_pattern = re.compile(r"Epoch (\d+)/\d+, Training Loss: (\d+\.\d+)")
    val_pattern = re.compile(r"Validation Accuracy: (\d+\.\d{4})")
    
    data = []
    current_epoch = None
    
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            
            # Buscar línea de entrenamiento
            train_match = train_pattern.search(line)
            if train_match:
                current_epoch = int(train_match.group(1))
                train_loss = float(train_match.group(2))
                data.append({"Epoch": current_epoch, "Training Loss": train_loss})
            
            # Buscar línea de validación (solo si ya encontramos una época)
            val_match = val_pattern.search(line)
            if val_match and current_epoch is not None:
                val_accuracy = float(val_match.group(1))
                # Actualizar la entrada correspondiente
                for entry in data:
                    if entry["Epoch"] == current_epoch:
                        entry["Validation Accuracy"] = val_accuracy
                        break
    
    return pd.DataFrame(data)

def main():
    # Parsear el archivo
    df = parse_training_log("results_training.txt")
    
    # Guardar en CSV
    df.to_csv("training_results.csv", index=False)
    print("Datos guardados en training_results.csv")
    
    # Crear gráficas
    plt.figure(figsize=(12, 6))
    
    # Gráfica de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(df["Epoch"], df["Training Loss"], "b-o")
    plt.title("Evolución de la Pérdida")
    plt.xlabel("Época")
    plt.ylabel("Training Loss")
    plt.grid(True)
    
    # Gráfica de precisión
    plt.subplot(1, 2, 2)
    plt.plot(df["Epoch"], df["Validation Accuracy"], "r-s")
    plt.title("Evolución de la Precisión")
    plt.xlabel("Época")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()

if __name__ == "__main__":
    main()