I have completed my DeepCos project, a cosmetic ingredient intelligence
application using CNN + OCR + Embedding + LSTM + MLP + MongoDB.

The functionality is already working.

DO NOT change the underlying ML models, prediction logic, database logic,
API contracts, training pipeline, or existing functionality.

I only want to simplify and improve the UI/UX.

The main goal is:

1. Make DeepCos understandable to a normal user.
2. Reduce technical/ML jargon from the normal interface.
3. Keep important technical information available for project demonstration.
4. Do not show every internal metric or implementation detail.
5. Use progressive disclosure: simple information first, technical details only
   when useful.
6. Do not remove existing model explainability.
7. Do not remove the knowledge base.
8. Do not remove model information.
9. Instead, simplify and reorganize them.

IMPORTANT:
Do not rewrite the whole frontend.
Modify the existing components and styling wherever possible.
Preserve the current dark/pink DeepCos visual identity.
Do not change backend APIs unless absolutely necessary.
STEP 1 — SIDEBAR

Change the main sidebar to only:

DeepCos
INGREDIENT
INTELLIGENCE

Dashboard
New Analysis
History
Ingredient Guide

Do NOT show:

Models
Knowledge Base
AI Enrichment
Database
MongoDB
CNN
LSTM
MLP

The user should not feel like they are using an ML developer dashboard.

STEP 2 — DASHBOARD

Remove the large technical status cards:

Ingredient Model
Label CNN
Knowledge Base
Storage

Do not show:

Embedding → LSTM → MLP
MongoDB
230 ingredients
25 functions
14 concern rules
OCR version
model status

Instead show:

DeepCos

Understand what's inside your cosmetic products.

[ + New Analysis ]

Then:

Recent Analyses

Show only:

Product type
Analysis summary
Number of ingredients
Date/time
Image/Text

Example:

Serum
Hydration-oriented with a secondary brightening profile.

12 ingredients
Image • Today

Keep the existing cards but make them smaller and cleaner.

STEP 3 — DASHBOARD "HOW IT WORKS"

Replace the three large technical cards.

Do NOT display:

AlexNet
OCR
Embedding
LSTM
MLP
Occlusion

in the main dashboard.

Instead show:

How DeepCos works

1. Read
Product labels are converted into ingredient information.

2. Analyze
DeepCos identifies patterns in the ingredient combination.

3. Explain
The result shows the formulation characteristics and
the ingredients that influenced the prediction.

This should be understandable to a non-technical user.

Optionally add:

Powered by computer vision and deep learning.

That's enough.

STEP 4 — NEW ANALYSIS PAGE

Make this extremely simple.

Title:

Analyze a Product

Subtitle:

Upload a product label or enter the ingredient list manually.

Two options:

┌───────────────────────┐
│                       │
│       Upload Image    │
│                       │
│   Product label       │
│                       │
└───────────────────────┘

            OR

┌───────────────────────────────┐
│ Paste ingredients here...     │
│                               │
└───────────────────────────────┘

       [ Analyze Product ]

Don't show model names or preprocessing information here.

STEP 5 — REPORT HEADER

Current technical information such as:

ID
12 ingredients
0.5s
Embedding → LSTM → MLP

should NOT appear in the main header.

Use:

Analysis Report

Likely Product Type
Exfoliant

12 ingredients analyzed

Optionally:

Analysis based on ingredient composition

That's enough.

STEP 6 — PRODUCT IDENTIFICATION

Replace:

47%
top-1 category probability
model argmax

with:

Likely Product Type

🧪 Exfoliant

Confidence: Moderate

If alternative categories are useful:

Other possible types
Serum
Toner

Do not display raw softmax terminology.

STEP 7 — FORMULATION PROFILE

Keep this section because it is one of the most important parts of DeepCos.

Use:

Formulation Profile

💧 Hydration
Moderate

✨ Brightening
Moderate

🧪 Exfoliation
High

🛡 Barrier Support
Moderate

Use simple visual bars.

Do NOT show:

R²
MAE
indicative
model regression
raw prediction values

Those belong in the simplified technical model page.

STEP 8 — SUMMARY

Keep a simple natural-language summary.

Example:

Summary

This formula shows characteristics associated with
exfoliation, with additional hydration-related properties.

Do not say:

The LSTM predicted...
The regression head produced...

The user doesn't need to know that here.

STEP 9 — KEY INGREDIENTS

Keep the current Key Ingredients section, but simplify it.

Instead of:

Importance 90.8
occlusion Δ 0.053

show:

Key Ingredients

Sodium Hyaluronate
Humectant
Helps attract and retain water.

Panthenol
Humectant · Soothing
Commonly associated with hydration and comfort.

Niacinamide
Brightening-associated · Skin conditioning

Remove raw importance numbers from the normal UI.

If desired, add:

Why is this important?

as a small expandable option.

STEP 10 — EXPLAINABILITY SECTION

Rename:

Why? (model explainability)

to:

Why these ingredients matter

Description:

DeepCos checks how individual ingredients influence each
formulation prediction.

Do NOT use:

pushed up
pushed down
Δ vs baseline

Instead use:

✓ Supports prediction

Sodium Hyaluronate
Niacinamide
Betaine
Aqua
Panthenol

and:

− Reduces prediction

Sodium Hydroxide
Allantoin

This applies separately to:

Hydration
Brightening
Exfoliation
Barrier Support
STEP 11 — REMOVE RAW EXPLAINABILITY NUMBERS FROM NORMAL VIEW

Do not show:

+0.011
+0.009
+0.005
-0.011

by default.

Add a small expandable control:

Show technical explanation

When clicked, show:

Occlusion-based attribution

Sodium Hyaluronate    +0.011
Niacinamide           +0.009
Betaine               +0.005
...

And explain:

These values represent the change in model output when
an individual ingredient is temporarily removed.

This preserves the actual ML explainability without overwhelming the user.

STEP 12 — IMPORTANT DISTINCTION

Do not confuse:

Ingredient function

with:

Model influence

For example:

Sodium Hyaluronate

Ingredient information:
Humectant
Water-binding

Model explanation:
Supports the hydration prediction

This is important because the knowledge base explains what the ingredient is, while the ML model explains how it influenced the prediction.

Keep both concepts separate.

STEP 13 — POTENTIAL CONCERNS

Rename:

Concern Screening

to:

Potential Concerns

Example:

Potential Concerns

ⓘ Preservative system detected

The formula contains ingredients commonly used
as preservatives.

This is informational and does not determine
personal suitability.

If nothing is found:

✓ No documented concern detected

No ingredient matched the available concern rules.

Do not use alarming wording such as:

Danger
Unsafe
Toxic
Harmful

unless the underlying evidence genuinely supports such a statement.

STEP 14 — FULL INGREDIENT LIST

Keep the complete ingredient section.

Use:

Ingredients (12)

Ingredient
Function
Information

Keep:

Aqua
Solvent

Glycerin
Humectant

Sodium Hyaluronate
Humectant

Niacinamide
Skin conditioning · Brightening-associated

Remove unnecessary technical metadata from the first view.

STEP 15 — DISCLAIMER

Have ONE concise disclaimer near the bottom.

Use:

ⓘ About this analysis

DeepCos provides educational information based on ingredient
composition and trained model predictions. It does not determine
personal safety or replace medical or dermatological advice.

Do not repeat the disclaimer in every section.

STEP 16 — INGREDIENT GUIDE

Rename the user-facing Knowledge Base to:

Ingredient Guide

This is what appears in the sidebar.

The user sees:

Ingredient Guide

Search an ingredient...

[ Search ]

When searching:

Niacinamide

show:

Niacinamide

What it is
Vitamin B3 derivative.

Functions
• Skin conditioning
• Brightening-associated
• Barrier support

DeepCos information
Used as an ingredient reference when analyzing
cosmetic formulations.

Keep it simple.

STEP 17 — DO NOT SHOW ALL KNOWLEDGE BASE COUNTS

Do NOT prominently show:

230 ingredients
25 functions
14 concern rules
161 aliases

Those are database statistics.

If you want to show something, use:

Ingredient reference

or:

DeepCos ingredient reference

That's enough.

STEP 18 — AI-ASSISTED KNOWLEDGE BASE

Do NOT show this on the normal Ingredient Guide:

AI-learned entries
Qwen
confidence high
promoted
unreviewed
Resolve
Remove

These are implementation/admin details.

Keep them accessible only from the technical area if needed.

Rename internally:

AI-assisted entries

instead of:

AI-learned entries

because the latter can imply the system independently established verified scientific facts.

STEP 19 — /knowledgebase

Create a lightweight technical Knowledge Base view.

When the user enters:

/knowledgebase

show:

Knowledge Base

DeepCos uses a structured ingredient reference alongside
the machine-learning models.

Reference information includes:

• Ingredient names
• Ingredient functions
• Ingredient aliases
• Potential concern rules
• Ingredient descriptions

Then optionally:

Reference coverage

Ingredients: 230
Functions: 25
Concern rules: 14

That's enough.

Do NOT dump all 230 ingredients.

Do NOT show every alias.

Do NOT show all AI-generated records.

Do NOT show database implementation details.

STEP 20 — /model

When entering:

/model

show a simple technical overview.

Title:

How DeepCos analyzes ingredients

First:

Ingredient Sequence Model

Ingredients
      ↓
Embedding
      ↓
LSTM
      ↓
Prediction Heads
      ↓
Product Type + Formulation Profiles

Then explain:

Embedding
Converts ingredient names into numerical representations.

LSTM
Learns patterns from the order and combination of ingredients.

Prediction heads
Generate product category and formulation-profile predictions.

This is much better than simply showing:

Embedding → LSTM → MLP

because your teacher can understand what each component does.

STEP 21 — /model MODEL PERFORMANCE

Show only the important metrics.

Model Performance

Product category classification
Accuracy: 91.8%
Macro F1: 91.8%

Then:

Formulation prediction

Hydration
R²: 0.32

Brightening
R²: 0.40

Exfoliation
R²: 0.56

Barrier Support
R²: 0.68

Do NOT show every training parameter.

Don't show:

optimizer
batch size
learning rate
sequence padding
embedding dimension
hidden units
dropout
epochs

unless specifically needed on a deeper technical page.

STEP 22 — /model CNN + OCR

Keep a small section:

Image Analysis

Product label image
       ↓
Image processing
       ↓
CNN-based text-region detection
       ↓
OCR
       ↓
Ingredient text

Explain:

The vision pipeline identifies the relevant text region
and OCR converts the ingredient label into text for analysis.

Don't overwhelm the teacher with CNN implementation details.

If asked, explain the architecture verbally.

STEP 23 — /model EXPLAINABILITY

Add:

Explainability

DeepCos uses ingredient-level occlusion analysis.

Each ingredient is temporarily removed from the input.
The prediction is recalculated and compared with the
original prediction.

This identifies ingredients that support or reduce
each predicted formulation profile.

This is enough technical depth for a project demo.

STEP 24 — TRAINING CURVES

You currently have training curves.

Keep them under:

/model

but put them near the bottom:

Training Behaviour

[ Training Loss Graph ]

[ Validation Loss Graph ]

Don't put them on the main report.

STEP 25 — CNN METRICS

Do NOT show:

100%
100%
100%
100%

as giant dashboard cards.

Instead:

Vision Pipeline

Text-region detection
Evaluated on the available image test set

Label category recognition
Evaluated on the available image test set

If your test-set metrics are meaningful and independently evaluated, show the actual values.

The key is: always show the test-set context, especially if a metric is 100%.

STEP 26 — SEARCH COMMANDS

Keep only three commands:

/model
/knowledgebase
/help

When /help is entered:

DeepCos Commands

/model
View a simplified technical overview.

/knowledgebase
View the ingredient knowledge system.

/help
Show available commands.

Don't create unnecessary commands.

STEP 27 — FINAL NORMAL USER FLOW

The normal user should experience:

Dashboard
     ↓
New Analysis
     ↓
Upload Image / Enter Ingredients
     ↓
Analyze
     ↓
Product Type
     ↓
Formulation Profile
     ↓
Summary
     ↓
Key Ingredients
     ↓
Potential Concerns
     ↓
Why these ingredients matter
     ↓
Full Ingredients

That's the entire user experience.

STEP 28 — FINAL TEACHER DEMO FLOW

During your project presentation:

First show the normal application
Dashboard
     ↓
New Analysis
     ↓
Upload/enter ingredients
     ↓
Report

Explain the result.

Then say:

"DeepCos also provides technical inspection of the underlying AI system."

Type:

/model

Show:

Embedding
↓
LSTM
↓
Prediction heads

91.8% category accuracy

R² values

CNN + OCR

Explainability

Then type:

/knowledgebase

Show:

Ingredient reference
Functions
Concern rules
Aliases

This gives the teacher enough technical evidence without turning your application into an admin dashboard.

STEP 29 — FINAL UI RULE

Use this rule throughout the project:

Level 1 — User information
What did DeepCos find?
Level 2 — Explanation
Why did DeepCos find it?
Level 3 — Technical details
How does the model work?

The first two should be visible normally.

The third should be accessible through:

/model

Similarly:

Level 1
What is this ingredient?
Level 2
What does it do?
Level 3
How is it stored / mapped / enriched?

The first two belong to Ingredient Guide.

The third belongs to /knowledgebase.

STEP 30 — THE MOST IMPORTANT THING: DON'T CHANGE YOUR ML

Tell your coding AI explicitly:

IMPORTANT IMPLEMENTATION CONSTRAINTS:

Do not retrain the model.

Do not change model architecture.

Do not change prediction calculations.

Do not change occlusion attribution calculations.

Do not change knowledge-base data.

Do not remove existing APIs.

Do not remove existing functionality.

Do not fabricate new metrics.

Do not replace actual model outputs with hardcoded values.

Only change:
- UI presentation
- wording
- section organization
- visibility
- expandable/collapsible sections
- navigation
- search command routing
- styling
Final structure I recommend
DEEPCOS
│
├── Dashboard
│
├── New Analysis
│
├── History
│
├── Ingredient Guide
│
└── Search
      │
      ├── /model
      │     ├── Model overview
      │     ├── How Embedding → LSTM works
      │     ├── Performance
      │     ├── CNN + OCR
      │     ├── Explainability
      │     └── Training curves
      │
      └── /knowledgebase
            ├── What the knowledge base contains
            ├── Ingredient reference
            ├── Functions
            ├── Concern rules
            └── Reference statistics

And the report itself:

Analysis Report
│
├── Likely Product Type
│
├── Formulation Profile
│
├── Summary
│
├── Key Ingredients
│
├── Potential Concerns
│
├── Why These Ingredients Matter
│      ├── Supports prediction
│      └── Reduces prediction
│
├── Full Ingredients
│
└── About This Analysis