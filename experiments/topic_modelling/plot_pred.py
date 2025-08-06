from grid_search import *
from dataclasses import asdict
class EvalPlot(Evaluator):

    # def evaluate(self, predictions: List[LabelledRef]):
    #     predictions = self._get_projection_of_labelled_refs(predictions)
    #     refs_in_pred = [lr.ref for lr in predictions]
    #     refs_in_gold = [lr.ref for lr in gold_standard]
    #     predictions_filtered = [lr for lr in predictions if lr.ref in refs_in_gold]
    #     gold_filtered = [lr for lr in gold_standard if lr.ref in refs_in_pred]
    #     result = self._compute_f1_score(gold_filtered, predictions_filtered)
    #     return result
    def get_table(self, predictions: List[LabelledRef], with_lengths=False):
        predictions = self._get_projection_of_labelled_refs(predictions)
        refs_in_pred = [lr.ref for lr in predictions]
        refs_in_gold = [lr.ref for lr in gold_standard]
        predictions_filtered = [lr for lr in predictions if lr.ref in refs_in_gold]
        gold_filtered = [lr for lr in gold_standard if lr.ref in refs_in_pred]

        table = []
        for g_lr, p_lr in zip(gold_filtered, predictions_filtered):
            text = get_ref_text_with_fallback(g_lr.ref, 'en')
            table.append({
                "Ref": g_lr.ref,
                "Text": text,
                "Gold Slugs": set(g_lr.slugs),
                "Predicted Slugs": set(p_lr.slugs),
                "False Positives": set(p_lr.slugs)-set(g_lr.slugs),
                "False Negatives": set(g_lr.slugs)-set(p_lr.slugs),
            })
            if with_lengths:
                table[-1]["Gold Slugs Length"] = len(g_lr.slugs)
                table[-1]["Predicted Slugs Length"] = len(p_lr.slugs)
                table[-1]["False Positives Length"] = len(set(p_lr.slugs)-set(g_lr.slugs))
                table[-1]["False Negatives Length"] = len(set(g_lr.slugs)-set(p_lr.slugs))
        return table
    def plot_table(self, predictions: List[LabelledRef], filename='output.csv', with_lengths=False):
        table = self.get_table(predictions, with_lengths)
        with open(filename, mode='w', newline='') as file:
            # Get the fieldnames from the first dictionary (keys)
            fieldnames = table[0].keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(table)

    def dump_predictions_jsonl(self, predictions: List[LabelledRef], out_path):
        with open(out_path, "w", encoding="utf-8") as fh:
            for lr in predictions:
                json_line = json.dumps(asdict(lr), ensure_ascii=False)
                fh.write(json_line + "\n")
        return out_path

if __name__ == '__main__':
    considered_labels = file_to_slugs("evaluation_data/all_slugs_in_training_set.csv")
    gold_standard = jsonl_to_labelled_refs("evaluation_data/revised_gold.jsonl")
    gold_standard = add_implied_toc_slugs(gold_standard, considered_labels)
    refs_to_evaluate = [labelled_ref.ref for labelled_ref in gold_standard]
    plot_evaluator = EvalPlot(gold_standard, considered_labels)
    hyperparameters = {'docs_num': 1000*10, 'above_mean_threshold_factor':  0.3, 'power_relevance_fun': 3}
    # predictor = VectorDBPredictor(".chromadb_openai", **hyperparameters)
    # predictions = predictor.predict(refs_to_evaluate)
    # plot_evaluator.plot_table(predictions)
    # print("Recall:", plot_evaluator._compute_total_recall(gold_standard, predictions))

    predictor = ContextVectorDBPredictor(".chromadb_openai", **hyperparameters)
    # refs_to_evaluate = ["Abudarham, Fasts, Prayers 35"]
    # predictor = PredictorWithCacheWrapper(predictor)
    predictions = predictor.predict(refs_to_evaluate)
    plot_evaluator.dump_predictions_jsonl(predictions, "evaluation_data/predictions.jsonl")
    plot_evaluator.plot_table(predictions, with_lengths=True)
    print("Recall:", plot_evaluator.get_total_recall(predictions))
    print("Precision:", plot_evaluator.get_total_precision(predictions))
    print(predictions)


