# Executive Summary: Bank Customer Churn Risk Scoring

**Purpose.** Identify retail-banking customers likely to leave *before* they do, so retention spending is targeted rather than broad.

**Data.** 10,000 customers; 20.4% churned.

**Result.** A Gradient Boosting model ranks customers by churn probability with ROC-AUC 0.87. About 79% of customers it flags as likely leavers actually leave (precision), and overall accuracy is 87%. It outperforms the simple baseline model (ROC-AUC 0.78).

**What drives churn.**
- *Age:* churn climbs from 12% (30s) to 56% (50s).
- *Number of products:* 2 products = 7.6% churn; 1 product = 27.7%; 3-4 products = 83-100%.
- *Activity:* inactive customers churn at 27% vs 14% for active ones.
- *Geography:* Germany 32% vs about 16% in France and Spain.

**Recommended actions.** (1) Target 40-60 year-olds with relationship outreach; (2) steer single-product customers toward a second product; (3) investigate why 3-4 product customers leave; (4) re-engage inactive members; (5) launch a Germany retention review.

**Tool.** A web dashboard lets staff score any customer, test "what if" changes (e.g., adding a product, re-activating the customer) and view the reasons behind each score, supporting transparency and regulatory review.

**Cautions.** The model uses a one-time snapshot of data; gender and geography influence scores, so fairness checks are advised before operational use.
