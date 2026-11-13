from src.utils import process_diabetes


class Metrics:

    def __init__(self, y_pred, y_test):
        self.y_pred = y_pred
        self.y_test = y_test
        self.vp_c1 = 0
        self.vn_c1 = 0
        self.fp_c1 = 0
        self.fn_c1 = 0
        self.vp_c0 = 0
        self.vn_c0 = 0
        self.fp_c0 = 0
        self.fn_c0 = 0

    def set_param_classe1(self):
        for yp, yt in zip(self.y_pred, self.y_test):
            if yp == 1 and yt == 1:
                self.vp_c1 = self.vp_c1 + 1
            if yp == 1 and yt == 0:
                self.fp_c1 = self.fp_c1 + 1
            if yp == 0 and yt == 1:
                self.fn_c1 = self.fn_c1 + 1
            if yp == 0 and yt == 0:
                self.vn_c1 = self.vn_c1 + 1

    def set_param_classe2(self):
        # Parâmetros considerando a classe 0 como positiva
        for yp, yt in zip(self.y_pred, self.y_test):
            if yp == 0 and yt == 0:
                self.vp_c0 = self.vp_c0 + 1
            if yp == 0 and yt == 1:
                self.fp_c0 = self.fp_c0 + 1
            if yp == 1 and yt == 0:
                self.fn_c0 = self.fn_c0 + 1
            if yp == 1 and yt == 1:
                self.vn_c0 = self.vn_c0 + 1

    def compute_acuraccy(self):
        total = self.vp_c1 + self.vn_c1 + self.fp_c1 + self.fn_c1
        return (self.vp_c1 + self.vn_c1) / total if total > 0 else 0

    def compute_recall_c1(self):
        denom = self.vp_c1 + self.fn_c1
        return self.vp_c1 / denom if denom > 0 else 0

    def compute_recall_c0(self):
        denom = self.vp_c0 + self.fn_c0
        return self.vp_c0 / denom if denom > 0 else 0

    def compute_precision_c1(self):
        denom = self.vp_c1 + self.fp_c1
        return self.vp_c1 / denom if denom > 0 else 0

    def compute_precision_c0(self):
        denom = self.vp_c0 + self.fp_c0
        return self.vp_c0 / denom if denom > 0 else 0

    def compute_f1_score_c1(self):
        p = self.compute_precision_c1()
        r = self.compute_recall_c1()
        return (2 * p * r) / (p + r) if (p + r) > 0 else 0

    def compute_f1_score_c0(self):
        p = self.compute_precision_c0()
        r = self.compute_recall_c0()
        return (2 * p * r) / (p + r) if (p + r) > 0 else 0

    def compute_confusion_matriz(self):
        # [[VN_c1, FP_c1],
        #  [FN_c1, VP_c1]]
        return [
            [self.vn_c1, self.fp_c1],
            [self.fn_c1, self.vp_c1]
        ]


y_pred, y_test = process_diabetes()
mt = Metrics(y_pred, y_test)
mt.set_param_classe1()
mt.set_param_classe2()

print("acurácia geral:", mt.compute_acuraccy())
print("recall classe 0:", mt.compute_recall_c0())
print("recall classe 1:", mt.compute_recall_c1())
print("precision classe 0:", mt.compute_precision_c0())
print("precision classe 1:", mt.compute_precision_c1())
print("F1-score classe 0:", mt.compute_f1_score_c0())
print("F1-score classe 1:", mt.compute_f1_score_c1())
print("Matriz de confusão:")
for linha in mt.compute_confusion_matriz():
    print(linha)