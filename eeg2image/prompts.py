"""Visual prompts and class descriptions for EEG motor imagery classes.

VISUAL_PROMPTS   — rich image-generation prompts for Approach 1 (text-guided)
CLASS_DESCRIPTIONS — NL EEG descriptions used to extract Kandinsky image
                     embedding targets for training Approach 2 (no paired
                     EEG-image data required).
"""

# ── Approach 1: visual prompts for the Kandinsky prior ────────────────────
VISUAL_PROMPTS = {
    0: (
        "Scientific medical illustration of a human brain viewed from above. "
        "Right hemisphere motor cortex glowing with electric blue neural activity, "
        "representing left hand motor imagery. Event-related desynchronization shown "
        "as dark suppression regions. White background, high detail, 4K medical diagram."
    ),
    1: (
        "Scientific medical illustration of a human brain viewed from above. "
        "Left hemisphere motor cortex (C3 region) glowing orange-red, representing "
        "right hand motor imagery. Contralateral sensorimotor desynchronization visible. "
        "White background, high detail, 4K medical diagram."
    ),
    2: (
        "Scientific medical illustration of a human brain viewed from above. "
        "Central midline vertex (Cz) glowing bright teal-green symmetrically, "
        "representing bilateral feet motor imagery in the supplementary motor area. "
        "White background, high detail, 4K medical diagram."
    ),
    3: (
        "Scientific medical illustration of a human brain viewed from above. "
        "Bilateral lateral sensorimotor cortex glowing golden-yellow, representing "
        "tongue motor imagery in the orofacial region. Symmetric lateral activation. "
        "White background, high detail, 4K medical diagram."
    ),
}

# ── Approach 2: descriptions for Kandinsky image embedding targets ─────────
CLASS_DESCRIPTIONS = {
    0: [
        "Left hand motor imagery. Event-related desynchronization in mu and beta bands over right sensorimotor cortex. Contralateral motor planning.",
        "Imagined left hand movement. Right hemisphere C4 electrode desynchronization. Suppressed mu rhythm over right central region.",
        "EEG left hand motor imagery. Right-lateralized sensorimotor desynchronization 8-30 Hz range.",
    ],
    1: [
        "Right hand motor imagery. Event-related desynchronization in mu and beta bands over left sensorimotor cortex. Contralateral motor planning.",
        "Imagined right hand movement. Left hemisphere C3 electrode desynchronization. Suppressed mu rhythm over left central region.",
        "EEG right hand motor imagery. Left-lateralized sensorimotor desynchronization 8-30 Hz range.",
    ],
    2: [
        "Both feet motor imagery. Bilateral desynchronization over central midline area Cz. Supplementary motor area activation for lower limb imagery.",
        "Imagined foot movement. Mu and beta suppression over vertex Cz. Bilateral foot movement imagery.",
        "EEG feet motor imagery. Midline central desynchronization in sensorimotor bands. Supplementary motor area.",
    ],
    3: [
        "Tongue motor imagery. Event-related desynchronization over lateral sensorimotor regions bilaterally. Orofacial motor imagery.",
        "Imagined tongue movement. Bilateral sensorimotor desynchronization. Face and tongue representation in motor cortex.",
        "EEG tongue motor imagery. Widespread sensorimotor desynchronization bilateral distribution. Lateral primary motor cortex.",
    ],
}

# Style prefix passed to the Kandinsky prior in Approach 2 generation
STYLE_PROMPT = (
    "Scientific brain MRI visualization, neural activity heatmap, "
    "motor cortex activation, medical illustration, professional diagram, "
    "clean white background, high detail"
)
