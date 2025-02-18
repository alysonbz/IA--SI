from src.utils import process_diabetes

class Metrics:

    def __init__(self, y_pred, y_test):
        self.vp_c1 = 0  
        self.vn_c1 = 0  
        self.fp_c1 = 0  
        self.fn_c1 = 0  
        self.vp_c0 = 0 
        self.vn_c0 = 0  
        self.fp_c0 = 0  
        self.fn_c0 = 0  
        self.y_pred = y_pred
        self.y_test = y_test

    def set_param_classe1(self):
        for yp, yt in zip(self.y_pred, self.y_test):
            if yp == 1 and yt == 1:
                self.vp_c1 += 1  
            elif yp == 1 and yt == 0:
                self.fp_c1 += 1  
            elif yp == 0 and yt == 1:
                self.fn_c1 += 1  
            elif yp == 0 and yt == 0:
                self.vn_c1 += 1  

    def set_param_classe2(self):
        for yp, yt in zip(self.y_pred, self.y_test):
            if yp == 0 and yt == 0:
                self.vp_c0 += 1 
            elif yp == 0 and yt == 1:
                self.fp_c0 += 1  
            elif yp == 1 and yt == 0:
                self.fn_c0 += 1 
            elif yp == 1 and yt == 1:
                self.vn_c0 += 1  

    def compute_accuracy(self):
        total = len(self.y_test)
        return (self.vp_c1 + self.vn_c1) / total

    def compute_recall_c1(self):
        # Recall classe 1 = VP / (VP + FN)
        return self.vp_c1 / (self.vp_c1 + self.fn_c1) if (self.vp_c1 + self.fn_c1) != 0 else 0

    def compute_recall_c0(self):
        return self.vp_c0 / (self.vp_c0 + self.fn_c0) if (self.vp_c0 + self.fn_c0) != 0 else 0

    def compute_precision_c1(self):
        return self.vp_c1 / (self.vp_c1 + self.fp_c1) if (self.vp_c1 + self.fp_c1) != 0 else 0

    def compute_precision_c0(self):
        return self.vp_c0 / (self.vp_c0 + self.fp_c0) if (self.vp_c0 + self.fp_c0) != 0 else 0

    def compute_f1_score_c1(self):
        precision = self.compute_precision_c1()
        recall = self.compute_recall_c1()
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    def compute_f1_score_c0(self):
        precision = self.compute_precision_c0()
        recall = self.compute_recall_c0()
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    def compute_confusion_matrix(self):
        return {
            'VP Classe 1': self.vp_c1, 'FN Classe 1': self.fn_c1, 'FP Classe 1': self.fp_c1, 'VN Classe 1': self.vn_c1,
            'VP Classe 0': self.vp_c0, 'FN Classe 0': self.fn_c0, 'FP Classe 0': self.fp_c0, 'VN Classe 0': self.vn_c0
        }


y_pred, y_test = process_diabetes()

mt = Metrics(y_pred, y_test)

mt.set_param_classe1()

mt.set_param_classe2()

print("Acurácia geral:", mt.compute_accuracy())
print("Recall classe 0:", mt.compute_recall_c0())
print("Recall classe 1:", mt.compute_recall_c1())
print("Precision classe 0:", mt.compute_precision_c0())
print("Precision classe 1:", mt.compute_precision_c1())
print("F1-Score classe 0:", mt.compute_f1_score_c0())
print("F1-Score classe 1:", mt.compute_f1_score_c1())
print("Matriz de confusão:", mt.compute_confusion_matrix())