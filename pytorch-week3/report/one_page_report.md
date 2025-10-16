# PyTorch Week 3 Project — One-Page Visual Report

## **1. ResNet-18 on CIFAR-10**

**Objective:** Implement ResNet-18 from scratch, achieve ≥80% test accuracy, visualize results.  

**Training/Validation Curves**  
![Loss/Accuracy Curves](../runs/cls/curves_cls.png)  
*Training and validation loss and accuracy over epochs.*

**Confusion Matrix**  
![Confusion Matrix](../runs/cls/confusion_matrix.png)  
*Normalized confusion matrix showing clear diagonal dominance.*

**Prediction Grids**  
Correct Predictions:  
![Correct Predictions](../runs/cls/preds_grid.png)  

Incorrect Predictions:  
![Incorrect Predictions](../runs/cls/miscls_grid.png)  

**Grad-CAM Heatmaps**  
![Grad-CAM Example](../runs/cls/gradcam_0.png)  
*Highlights regions used by the model to make predictions.*

---

## **2. Transformer Encoder-Decoder on Toy Translation**

**Objective:** Implement Transformer from scratch, achieve BLEU ≥15, visualize attention and masks.  

**Training Curve (Loss)**  
![Loss Curve](../runs/mt/curves_mt.png)  
*Training loss over epochs.*

**Attention Heatmaps**  
![Attention Head Example](../runs/mt/attention_layer1_head1.png)  
*Attention patterns showing alignment between source and target tokens.*

**Mask Visualization**  
![Masks Demo](../runs/mt/masks_demo.png)  
*Demonstrates causal masking in decoder.*

**Decoded Outputs vs. Ground Truth**  
![Decoded Table](../runs/mt/decodes_table.png)  
*Comparison of generated outputs with target sequences.*

**BLEU Score Summary**  
![BLEU Score](../runs/mt/bleu_report.png)  
*Corpus BLEU score indicating translation quality.*

---

## **Key Insights**

- **ResNet-18:** Residual connections stabilize training and improve accuracy. Grad-CAM helps interpret predictions.  
- **Transformer:** Multi-head attention captures token dependencies; masking is essential for correct decoding.  
- **General:** Implementing architectures from primitives deepens understanding of forward/backward passes and model internals.

---

## **Challenges & Solutions**

| Challenge | Solution |
|-----------|---------|
| Residual shortcut dimensions mismatch | Used 1×1 conv projection |
| Transformer masking & attention | Implemented causal and padding masks |
| Training instability | Tuned learning rate, batch size, optimizer |
| Visualizations | Used matplotlib, seaborn, and Grad-CAM |

---

## **Repository Structure**

