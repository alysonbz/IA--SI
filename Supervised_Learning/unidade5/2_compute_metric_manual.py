from src.utils import process_diabetes


class Metrics:

    def __init__(self, y_pred, y_test):
        self.y_pred = y_pred
        self.y_test = y_test

        # Inicialização das métricas
        self.vp_c1 = 0  # Verdadeiros Positivos Classe 1
        self.vn_c1 = 0  # Verdadeiros Negativos Classe 1
        self.fp_c1 = 0  # Falsos Positivos Classe 1
        self.fn_c1 = 0  # Falsos Negativos Classe 1
        self.vp_c0 = 0  # Verdadeiros Positivos Classe 0
        self.vn_c0 = 0  # Verdadeiros Negativos Classe 0
        self.fp_c0 = 0  # Falsos Positivos Classe 0
        self.fn_c0 = 0  # Falsos Negativos Classe 0

    def set_param_classe1(self):
        """Calcula VP, VN, FP, FN para a classe 1."""
        for yp, yt in zip(self.y_pred, self.y_test):
            if yp == 1 and yt == 1:
                self.vp_c1 += 1
            if yp == 1 and yt == 0:
                self.fp_c1 += 1
            if yp == 0 and yt == 1:
                self.fn_c1 += 1
            if yp == 0 and yt == 0:
                self.vn_c1 += 1

    def set_param_classe2(self):
        """Calcula VP, VN, FP, FN para a classe 0 (equivalente à classe negativa)."""
        self.vp_c0 = self.vn_c1  # Verdadeiros positivos classe 0 = VN da classe 1
        self.vn_c0 = self.vp_c1  # Verdadeiros negativos classe 0 = VP da classe 1
        self.fp_c0 = self.fn_c1  # Falsos positivos classe 0 = FN da classe 1
        self.fn_c0 = self.fp_c1  # Falsos negativos classe 0 = FP da classe 1

    def compute_accuracy(self):
        total = self.vp_c1 + self.vn_c1 + self.fp_c1 + self.fn_c1
        return (self.vp_c1 + self.vn_c1) / total if total > 0 else 0

    def compute_recall_c1(self):
        return self.vp_c1 / (self.vp_c1 + self.fn_c1) if (self.vp_c1 + self.fn_c1) > 0 else 0

    def compute_recall_c0(self):
        return self.vp_c0 / (self.vp_c0 + self.fn_c0) if (self.vp_c0 + self.fn_c0) > 0 else 0

    def compute_precision_c1(self):
        return self.vp_c1 / (self.vp_c1 + self.fp_c1) if (self.vp_c1 + self.fp_c1) > 0 else 0

    def compute_precision_c0(self):
        return self.vp_c0 / (self.vp_c0 + self.fp_c0) if (self.vp_c0 + self.fp_c0) > 0 else 0

    def compute_f1_score_c1(self):
        precision = self.compute_precision_c1()
        recall = self.compute_recall_c1()
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def compute_f1_score_c0(self):
        precision = self.compute_precision_c0()
        recall = self.compute_recall_c0()
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def compute_confusion_matrix(self):
        return [[self.vn_c1, self.fp_c1], [self.fn_c1, self.vp_c1]]


# Processa os dados e cria a instância da classe
y_pred, y_test = process_diabetes()
mt = Metrics(y_pred, y_test)

# Calcula os parâmetros das classes
mt.set_param_classe1()
mt.set_param_classe2()

# Exibe os resultados
print("Acurácia geral:", mt.compute_accuracy())
print("Recall classe 0:", mt.compute_recall_c0())
print("Recall classe 1:", mt.compute_recall_c1())
print("Precisão classe 0:", mt.compute_precision_c0())
print("Precisão classe 1:", mt.compute_precision_c1())
print("F1-score classe 0:", mt.compute_f1_score_c0())
print("F1-score classe 1:", mt.compute_f1_score_c1())
print("Matriz de confusão:")
for row in mt.compute_confusion_matrix():
    print(row)
