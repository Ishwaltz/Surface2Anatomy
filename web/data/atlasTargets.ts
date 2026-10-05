export interface AtlasTargetDef {
  id: string;
  slot: number;
  name: string;
  category: string;
  system: string;
  coords: [number, number, number];
  color: number;
  hex: string;
  side: string;
  synonyms: string[];
}

export const ALL_ATLAS_TARGETS: AtlasTargetDef[] = [
  {
    "id": "brain",
    "slot": 1,
    "name": "Brain",
    "category": "Cranial & Neuroanatomy",
    "system": "cranial",
    "coords": [
      -2.9,
      52.0,
      350.0
    ],
    "color": 440020,
    "hex": "#06b6d4",
    "side": "anterior",
    "synonyms": [
      "cerebrum",
      "encephalon",
      "cranial cavity",
      "brainstem"
    ]
  },
  {
    "id": "skull",
    "slot": 2,
    "name": "Skull",
    "category": "Cranial & Neuroanatomy",
    "system": "cranial",
    "coords": [
      0.82,
      110.86,
      320.0
    ],
    "color": 440020,
    "hex": "#06b6d4",
    "side": "superior",
    "synonyms": [
      "cranium",
      "calvarium",
      "neurocranium"
    ]
  },
  {
    "id": "heart",
    "slot": 3,
    "name": "Heart",
    "category": "Thoracic & Respiratory",
    "system": "thoracic",
    "coords": [
      -25.8,
      85.0,
      75.8
    ],
    "color": 16007006,
    "hex": "#f43f5e",
    "side": "anterior",
    "synonyms": [
      "cardiac organ",
      "cor",
      "myocardium"
    ]
  },
  {
    "id": "trachea",
    "slot": 4,
    "name": "Trachea",
    "category": "Thoracic & Respiratory",
    "system": "thoracic",
    "coords": [
      3.22,
      52.78,
      164.39
    ],
    "color": 16007006,
    "hex": "#f43f5e",
    "side": "anterior",
    "synonyms": [
      "windpipe",
      "respiratory tube"
    ]
  },
  {
    "id": "esophagus",
    "slot": 5,
    "name": "Esophagus",
    "category": "Thoracic & Respiratory",
    "system": "thoracic",
    "coords": [
      -8.5,
      42.87,
      99.79
    ],
    "color": 16007006,
    "hex": "#f43f5e",
    "side": "posterior",
    "synonyms": [
      "gullet",
      "alimentary canal"
    ]
  },
  {
    "id": "thyroid_gland",
    "slot": 6,
    "name": "Thyroid Gland",
    "category": "Thoracic & Respiratory",
    "system": "thoracic",
    "coords": [
      1.4,
      79.49,
      215.77
    ],
    "color": 16007006,
    "hex": "#f43f5e",
    "side": "anterior",
    "synonyms": [
      "thyroid",
      "glandula thyroidea"
    ]
  },
  {
    "id": "lung_upper_lobe_left",
    "slot": 7,
    "name": "Left Upper Lung Lobe",
    "category": "Thoracic & Respiratory",
    "system": "thoracic",
    "coords": [
      -79.44,
      72.83,
      116.58
    ],
    "color": 16007006,
    "hex": "#f43f5e",
    "side": "anterior",
    "synonyms": [
      "LUL",
      "left upper lung"
    ]
  },
  {
    "id": "lung_lower_lobe_left",
    "slot": 8,
    "name": "Left Lower Lung Lobe",
    "category": "Thoracic & Respiratory",
    "system": "thoracic",
    "coords": [
      -73.55,
      11.25,
      74.61
    ],
    "color": 16007006,
    "hex": "#f43f5e",
    "side": "posterior",
    "synonyms": [
      "LLL",
      "left lower lung"
    ]
  },
  {
    "id": "lung_upper_lobe_right",
    "slot": 9,
    "name": "Right Upper Lung Lobe",
    "category": "Thoracic & Respiratory",
    "system": "thoracic",
    "coords": [
      56.72,
      67.26,
      137.4
    ],
    "color": 16007006,
    "hex": "#f43f5e",
    "side": "anterior",
    "synonyms": [
      "RUL",
      "right upper lung"
    ]
  },
  {
    "id": "lung_middle_lobe_right",
    "slot": 10,
    "name": "Right Middle Lung Lobe",
    "category": "Thoracic & Respiratory",
    "system": "thoracic",
    "coords": [
      71.21,
      99.15,
      80.52
    ],
    "color": 16007006,
    "hex": "#f43f5e",
    "side": "anterior",
    "synonyms": [
      "RML",
      "right middle lung"
    ]
  },
  {
    "id": "lung_lower_lobe_right",
    "slot": 11,
    "name": "Right Lower Lung Lobe",
    "category": "Thoracic & Respiratory",
    "system": "thoracic",
    "coords": [
      66.74,
      14.55,
      74.16
    ],
    "color": 16007006,
    "hex": "#f43f5e",
    "side": "posterior",
    "synonyms": [
      "RLL",
      "right lower lung"
    ]
  },
  {
    "id": "aorta",
    "slot": 12,
    "name": "Aorta",
    "category": "Cardiovascular & Major Vessels",
    "system": "vascular",
    "coords": [
      -12.0,
      28.0,
      45.0
    ],
    "color": 15680580,
    "hex": "#ef4444",
    "side": "posterior",
    "synonyms": [
      "thoracic aorta",
      "abdominal aorta"
    ]
  },
  {
    "id": "superior_vena_cava",
    "slot": 13,
    "name": "Superior Vena Cava",
    "category": "Cardiovascular & Major Vessels",
    "system": "vascular",
    "coords": [
      24.87,
      68.95,
      127.96
    ],
    "color": 15680580,
    "hex": "#ef4444",
    "side": "anterior",
    "synonyms": [
      "SVC",
      "precava"
    ]
  },
  {
    "id": "inferior_vena_cava",
    "slot": 14,
    "name": "Inferior Vena Cava",
    "category": "Cardiovascular & Major Vessels",
    "system": "vascular",
    "coords": [
      20.29,
      58.93,
      -11.41
    ],
    "color": 15680580,
    "hex": "#ef4444",
    "side": "posterior",
    "synonyms": [
      "IVC",
      "postcava"
    ]
  },
  {
    "id": "pulmonary_vein",
    "slot": 15,
    "name": "Pulmonary Vein",
    "category": "Cardiovascular & Major Vessels",
    "system": "vascular",
    "coords": [
      -2.54,
      40.66,
      97.75
    ],
    "color": 15680580,
    "hex": "#ef4444",
    "side": "posterior",
    "synonyms": [
      "venae pulmonales"
    ]
  },
  {
    "id": "brachiocephalic_trunk",
    "slot": 16,
    "name": "Brachiocephalic Trunk",
    "category": "Cardiovascular & Major Vessels",
    "system": "vascular",
    "coords": [
      6.42,
      76.85,
      176.12
    ],
    "color": 15680580,
    "hex": "#ef4444",
    "side": "anterior",
    "synonyms": [
      "innominate artery"
    ]
  },
  {
    "id": "subclavian_artery_right",
    "slot": 17,
    "name": "Right Subclavian Artery",
    "category": "Cardiovascular & Major Vessels",
    "system": "vascular",
    "coords": [
      43.54,
      61.42,
      201.01
    ],
    "color": 15680580,
    "hex": "#ef4444",
    "side": "anterior",
    "synonyms": [
      "right subclavian"
    ]
  },
  {
    "id": "subclavian_artery_left",
    "slot": 18,
    "name": "Left Subclavian Artery",
    "category": "Cardiovascular & Major Vessels",
    "system": "vascular",
    "coords": [
      -38.7,
      57.55,
      195.23
    ],
    "color": 15680580,
    "hex": "#ef4444",
    "side": "anterior",
    "synonyms": [
      "left subclavian"
    ]
  },
  {
    "id": "common_carotid_artery_right",
    "slot": 19,
    "name": "Right Common Carotid",
    "category": "Cardiovascular & Major Vessels",
    "system": "vascular",
    "coords": [
      20.59,
      72.5,
      213.59
    ],
    "color": 15680580,
    "hex": "#ef4444",
    "side": "anterior",
    "synonyms": [
      "right carotid"
    ]
  },
  {
    "id": "common_carotid_artery_left",
    "slot": 20,
    "name": "Left Common Carotid",
    "category": "Cardiovascular & Major Vessels",
    "system": "vascular",
    "coords": [
      -16.71,
      69.58,
      199.82
    ],
    "color": 15680580,
    "hex": "#ef4444",
    "side": "anterior",
    "synonyms": [
      "left carotid"
    ]
  },
  {
    "id": "brachiocephalic_vein_left",
    "slot": 21,
    "name": "Left Brachiocephalic Vein",
    "category": "Cardiovascular & Major Vessels",
    "system": "vascular",
    "coords": [
      -6.53,
      84.29,
      173.6
    ],
    "color": 15680580,
    "hex": "#ef4444",
    "side": "anterior",
    "synonyms": [
      "left innominate vein"
    ]
  },
  {
    "id": "brachiocephalic_vein_right",
    "slot": 22,
    "name": "Right Brachiocephalic Vein",
    "category": "Cardiovascular & Major Vessels",
    "system": "vascular",
    "coords": [
      29.13,
      74.42,
      178.37
    ],
    "color": 15680580,
    "hex": "#ef4444",
    "side": "anterior",
    "synonyms": [
      "right innominate vein"
    ]
  },
  {
    "id": "atrial_appendage_left",
    "slot": 23,
    "name": "Left Atrial Appendage",
    "category": "Cardiovascular & Major Vessels",
    "system": "vascular",
    "coords": [
      -35.67,
      63.23,
      103.21
    ],
    "color": 15680580,
    "hex": "#ef4444",
    "side": "anterior",
    "synonyms": [
      "left auricle"
    ]
  },
  {
    "id": "portal_vein_and_splenic_vein",
    "slot": 24,
    "name": "Portal & Splenic Vein",
    "category": "Cardiovascular & Major Vessels",
    "system": "vascular",
    "coords": [
      6.16,
      73.71,
      -14.51
    ],
    "color": 15680580,
    "hex": "#ef4444",
    "side": "posterior",
    "synonyms": [
      "portal venous confluence"
    ]
  },
  {
    "id": "iliac_artery_left",
    "slot": 25,
    "name": "Left Iliac Artery",
    "category": "Cardiovascular & Major Vessels",
    "system": "vascular",
    "coords": [
      -40.0,
      67.58,
      -179.06
    ],
    "color": 15680580,
    "hex": "#ef4444",
    "side": "anterior",
    "synonyms": [
      "left common iliac artery"
    ]
  },
  {
    "id": "iliac_artery_right",
    "slot": 26,
    "name": "Right Iliac Artery",
    "category": "Cardiovascular & Major Vessels",
    "system": "vascular",
    "coords": [
      37.65,
      70.07,
      -181.37
    ],
    "color": 15680580,
    "hex": "#ef4444",
    "side": "anterior",
    "synonyms": [
      "right common iliac artery"
    ]
  },
  {
    "id": "iliac_vena_left",
    "slot": 27,
    "name": "Left Iliac Vein",
    "category": "Cardiovascular & Major Vessels",
    "system": "vascular",
    "coords": [
      -35.4,
      57.78,
      -187.69
    ],
    "color": 15680580,
    "hex": "#ef4444",
    "side": "posterior",
    "synonyms": [
      "left common iliac vein"
    ]
  },
  {
    "id": "iliac_vena_right",
    "slot": 28,
    "name": "Right Iliac Vein",
    "category": "Cardiovascular & Major Vessels",
    "system": "vascular",
    "coords": [
      40.96,
      60.56,
      -191.55
    ],
    "color": 15680580,
    "hex": "#ef4444",
    "side": "posterior",
    "synonyms": [
      "right common iliac vein"
    ]
  },
  {
    "id": "liver",
    "slot": 29,
    "name": "Liver",
    "category": "Abdominal & Digestive",
    "system": "abdominal",
    "coords": [
      63.1,
      72.0,
      5.2
    ],
    "color": 1096065,
    "hex": "#10b981",
    "side": "anterior",
    "synonyms": [
      "hepatic parenchyma",
      "hepar"
    ]
  },
  {
    "id": "spleen",
    "slot": 30,
    "name": "Spleen",
    "category": "Abdominal & Digestive",
    "system": "abdominal",
    "coords": [
      -94.56,
      32.0,
      0.4
    ],
    "color": 1096065,
    "hex": "#10b981",
    "side": "lateral",
    "synonyms": [
      "lien",
      "splenic tissue"
    ]
  },
  {
    "id": "pancreas",
    "slot": 31,
    "name": "Pancreas",
    "category": "Abdominal & Digestive",
    "system": "abdominal",
    "coords": [
      8.0,
      50.0,
      15.0
    ],
    "color": 1096065,
    "hex": "#10b981",
    "side": "anterior",
    "synonyms": [
      "pancreatic gland",
      "pancreatic body"
    ]
  },
  {
    "id": "gallbladder",
    "slot": 32,
    "name": "Gallbladder",
    "category": "Abdominal & Digestive",
    "system": "abdominal",
    "coords": [
      69.49,
      99.77,
      -32.44
    ],
    "color": 1096065,
    "hex": "#10b981",
    "side": "anterior",
    "synonyms": [
      "cholecyst",
      "biliary vesicle"
    ]
  },
  {
    "id": "stomach",
    "slot": 33,
    "name": "Stomach",
    "category": "Abdominal & Digestive",
    "system": "abdominal",
    "coords": [
      -45.52,
      90.24,
      -1.04
    ],
    "color": 1096065,
    "hex": "#10b981",
    "side": "anterior",
    "synonyms": [
      "gaster",
      "ventriculus"
    ]
  },
  {
    "id": "duodenum",
    "slot": 34,
    "name": "Duodenum",
    "category": "Abdominal & Digestive",
    "system": "abdominal",
    "coords": [
      24.45,
      76.93,
      -58.55
    ],
    "color": 1096065,
    "hex": "#10b981",
    "side": "anterior",
    "synonyms": [
      "proximal small intestine"
    ]
  },
  {
    "id": "small_bowel",
    "slot": 35,
    "name": "Small Bowel",
    "category": "Abdominal & Digestive",
    "system": "abdominal",
    "coords": [
      -24.46,
      91.81,
      -113.43
    ],
    "color": 1096065,
    "hex": "#10b981",
    "side": "anterior",
    "synonyms": [
      "jejunum",
      "ileum",
      "small intestine"
    ]
  },
  {
    "id": "colon",
    "slot": 36,
    "name": "Colon",
    "category": "Abdominal & Digestive",
    "system": "abdominal",
    "coords": [
      -9.61,
      83.68,
      -94.81
    ],
    "color": 1096065,
    "hex": "#10b981",
    "side": "anterior",
    "synonyms": [
      "large bowel",
      "large intestine"
    ]
  },
  {
    "id": "kidney_right",
    "slot": 37,
    "name": "Right Kidney",
    "category": "Abdominal & Digestive",
    "system": "abdominal",
    "coords": [
      68.5,
      25.0,
      -49.5
    ],
    "color": 1096065,
    "hex": "#10b981",
    "side": "posterior",
    "synonyms": [
      "right renal gland",
      "ren dexter"
    ]
  },
  {
    "id": "kidney_left",
    "slot": 38,
    "name": "Left Kidney",
    "category": "Abdominal & Digestive",
    "system": "abdominal",
    "coords": [
      -68.5,
      25.0,
      -49.5
    ],
    "color": 1096065,
    "hex": "#10b981",
    "side": "posterior",
    "synonyms": [
      "left renal gland",
      "ren sinister"
    ]
  },
  {
    "id": "adrenal_gland_right",
    "slot": 39,
    "name": "Right Adrenal Gland",
    "category": "Abdominal & Digestive",
    "system": "abdominal",
    "coords": [
      32.85,
      36.03,
      -9.98
    ],
    "color": 1096065,
    "hex": "#10b981",
    "side": "posterior",
    "synonyms": [
      "right suprarenal gland"
    ]
  },
  {
    "id": "adrenal_gland_left",
    "slot": 40,
    "name": "Left Adrenal Gland",
    "category": "Abdominal & Digestive",
    "system": "abdominal",
    "coords": [
      -36.97,
      44.72,
      -17.72
    ],
    "color": 1096065,
    "hex": "#10b981",
    "side": "posterior",
    "synonyms": [
      "left suprarenal gland"
    ]
  },
  {
    "id": "kidney_cyst_left",
    "slot": 41,
    "name": "Left Kidney Cyst",
    "category": "Abdominal & Digestive",
    "system": "abdominal",
    "coords": [
      -77.0,
      22.57,
      -58.0
    ],
    "color": 1096065,
    "hex": "#10b981",
    "side": "posterior",
    "synonyms": [
      "left renal cyst"
    ]
  },
  {
    "id": "kidney_cyst_right",
    "slot": 42,
    "name": "Right Kidney Cyst",
    "category": "Abdominal & Digestive",
    "system": "abdominal",
    "coords": [
      66.75,
      28.72,
      -80.09
    ],
    "color": 1096065,
    "hex": "#10b981",
    "side": "posterior",
    "synonyms": [
      "right renal cyst"
    ]
  },
  {
    "id": "urinary_bladder",
    "slot": 43,
    "name": "Urinary Bladder",
    "category": "Pelvis, Reproductive & Lower Girdle",
    "system": "pelvis",
    "coords": [
      0.0,
      48.0,
      -95.0
    ],
    "color": 14239471,
    "hex": "#d946ef",
    "side": "anterior",
    "synonyms": [
      "vesica urinaria",
      "bladder"
    ]
  },
  {
    "id": "prostate",
    "slot": 44,
    "name": "Prostate Gland",
    "category": "Pelvis, Reproductive & Lower Girdle",
    "system": "pelvis",
    "coords": [
      0.0,
      38.0,
      -105.0
    ],
    "color": 14239471,
    "hex": "#d946ef",
    "side": "anterior",
    "synonyms": [
      "prostatic gland",
      "male reproductive"
    ]
  },
  {
    "id": "uterus",
    "slot": 45,
    "name": "Uterus",
    "category": "Pelvis, Reproductive & Lower Girdle",
    "system": "pelvis",
    "coords": [
      0.0,
      42.0,
      -78.0
    ],
    "color": 14239471,
    "hex": "#d946ef",
    "side": "anterior",
    "synonyms": [
      "womb",
      "uterine body",
      "myometrium",
      "female pelvis"
    ]
  },
  {
    "id": "ovary_left",
    "slot": 46,
    "name": "Left Ovary",
    "category": "Pelvis, Reproductive & Lower Girdle",
    "system": "pelvis",
    "coords": [
      -42.0,
      38.0,
      -75.0
    ],
    "color": 14239471,
    "hex": "#d946ef",
    "side": "lateral",
    "synonyms": [
      "left adnexa",
      "ovarium sinistrum",
      "female gonad"
    ]
  },
  {
    "id": "ovary_right",
    "slot": 47,
    "name": "Right Ovary",
    "category": "Pelvis, Reproductive & Lower Girdle",
    "system": "pelvis",
    "coords": [
      42.0,
      38.0,
      -75.0
    ],
    "color": 14239471,
    "hex": "#d946ef",
    "side": "lateral",
    "synonyms": [
      "right adnexa",
      "ovarium dextrum",
      "female gonad"
    ]
  },
  {
    "id": "vagina",
    "slot": 48,
    "name": "Vagina",
    "category": "Pelvis, Reproductive & Lower Girdle",
    "system": "pelvis",
    "coords": [
      0.0,
      32.0,
      -125.0
    ],
    "color": 14239471,
    "hex": "#d946ef",
    "side": "anterior",
    "synonyms": [
      "vaginal canal",
      "female birth canal",
      "colpos"
    ]
  },
  {
    "id": "sacrum",
    "slot": 49,
    "name": "Sacrum",
    "category": "Pelvis, Reproductive & Lower Girdle",
    "system": "pelvis",
    "coords": [
      0.55,
      -14.18,
      -181.08
    ],
    "color": 14239471,
    "hex": "#d946ef",
    "side": "posterior",
    "synonyms": [
      "sacral bone",
      "sacral vertebra"
    ]
  },
  {
    "id": "hip_left",
    "slot": 50,
    "name": "Left Hip (Os Coxae)",
    "category": "Pelvis, Reproductive & Lower Girdle",
    "system": "pelvis",
    "coords": [
      -72.5,
      28.57,
      -204.62
    ],
    "color": 14239471,
    "hex": "#d946ef",
    "side": "lateral",
    "synonyms": [
      "left ilium",
      "left pelvic bone"
    ]
  },
  {
    "id": "hip_right",
    "slot": 51,
    "name": "Right Hip (Os Coxae)",
    "category": "Pelvis, Reproductive & Lower Girdle",
    "system": "pelvis",
    "coords": [
      72.56,
      29.1,
      -205.52
    ],
    "color": 14239471,
    "hex": "#d946ef",
    "side": "lateral",
    "synonyms": [
      "right ilium",
      "right pelvic bone"
    ]
  },
  {
    "id": "femur_left",
    "slot": 52,
    "name": "Left Femur Head",
    "category": "Pelvis, Reproductive & Lower Girdle",
    "system": "pelvis",
    "coords": [
      -105.86,
      36.73,
      -282.31
    ],
    "color": 14239471,
    "hex": "#d946ef",
    "side": "lateral",
    "synonyms": [
      "left thigh bone"
    ]
  },
  {
    "id": "femur_right",
    "slot": 53,
    "name": "Right Femur Head",
    "category": "Pelvis, Reproductive & Lower Girdle",
    "system": "pelvis",
    "coords": [
      104.75,
      37.22,
      -285.03
    ],
    "color": 14239471,
    "hex": "#d946ef",
    "side": "lateral",
    "synonyms": [
      "right thigh bone"
    ]
  },
  {
    "id": "gluteus_maximus_left",
    "slot": 54,
    "name": "Left Gluteus Maximus",
    "category": "Pelvis, Reproductive & Lower Girdle",
    "system": "pelvis",
    "coords": [
      -87.87,
      -20.57,
      -247.64
    ],
    "color": 14239471,
    "hex": "#d946ef",
    "side": "posterior",
    "synonyms": [
      "left upper buttock muscle"
    ]
  },
  {
    "id": "gluteus_maximus_right",
    "slot": 55,
    "name": "Right Gluteus Maximus",
    "category": "Pelvis, Reproductive & Lower Girdle",
    "system": "pelvis",
    "coords": [
      87.55,
      -20.41,
      -250.41
    ],
    "color": 14239471,
    "hex": "#d946ef",
    "side": "posterior",
    "synonyms": [
      "right upper buttock muscle"
    ]
  },
  {
    "id": "gluteus_medius_left",
    "slot": 56,
    "name": "Left Gluteus Medius",
    "category": "Pelvis, Reproductive & Lower Girdle",
    "system": "pelvis",
    "coords": [
      -115.72,
      24.17,
      -184.41
    ],
    "color": 14239471,
    "hex": "#d946ef",
    "side": "lateral",
    "synonyms": [
      "left middle buttock muscle"
    ]
  },
  {
    "id": "gluteus_medius_right",
    "slot": 57,
    "name": "Right Gluteus Medius",
    "category": "Pelvis, Reproductive & Lower Girdle",
    "system": "pelvis",
    "coords": [
      117.53,
      24.74,
      -191.72
    ],
    "color": 14239471,
    "hex": "#d946ef",
    "side": "lateral",
    "synonyms": [
      "right middle buttock muscle"
    ]
  },
  {
    "id": "gluteus_minimus_left",
    "slot": 58,
    "name": "Left Gluteus Minimus",
    "category": "Pelvis, Reproductive & Lower Girdle",
    "system": "pelvis",
    "coords": [
      -112.01,
      47.58,
      -210.22
    ],
    "color": 14239471,
    "hex": "#d946ef",
    "side": "lateral",
    "synonyms": [
      "left deep buttock muscle"
    ]
  },
  {
    "id": "gluteus_minimus_right",
    "slot": 59,
    "name": "Right Gluteus Minimus",
    "category": "Pelvis, Reproductive & Lower Girdle",
    "system": "pelvis",
    "coords": [
      111.84,
      46.22,
      -213.73
    ],
    "color": 14239471,
    "hex": "#d946ef",
    "side": "lateral",
    "synonyms": [
      "right deep buttock muscle"
    ]
  },
  {
    "id": "iliopsoas_left",
    "slot": 60,
    "name": "Left Iliopsoas",
    "category": "Pelvis, Reproductive & Lower Girdle",
    "system": "pelvis",
    "coords": [
      -60.06,
      44.67,
      -146.73
    ],
    "color": 14239471,
    "hex": "#d946ef",
    "side": "anterior",
    "synonyms": [
      "left psoas hip flexor"
    ]
  },
  {
    "id": "iliopsoas_right",
    "slot": 61,
    "name": "Right Iliopsoas",
    "category": "Pelvis, Reproductive & Lower Girdle",
    "system": "pelvis",
    "coords": [
      56.94,
      42.36,
      -146.44
    ],
    "color": 14239471,
    "hex": "#d946ef",
    "side": "anterior",
    "synonyms": [
      "right psoas hip flexor"
    ]
  },
  {
    "id": "spinal_cord",
    "slot": 62,
    "name": "Spinal Cord",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      -0.15,
      1.29,
      34.62
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "medulla spinalis",
      "spinal canal"
    ]
  },
  {
    "id": "vertebrae_C1",
    "slot": 63,
    "name": "Vertebra C1 (Atlas)",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      -3.17,
      63.57,
      280.3
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "atlas vertebra"
    ]
  },
  {
    "id": "vertebrae_C2",
    "slot": 64,
    "name": "Vertebra C2 (Axis)",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      -3.06,
      61.56,
      270.04
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "axis vertebra"
    ]
  },
  {
    "id": "vertebrae_C3",
    "slot": 65,
    "name": "Vertebra C3",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      -2.04,
      62.38,
      270.19
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "third cervical vertebra"
    ]
  },
  {
    "id": "vertebrae_C4",
    "slot": 66,
    "name": "Vertebra C4",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      -1.72,
      59.74,
      253.25
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "fourth cervical vertebra"
    ]
  },
  {
    "id": "vertebrae_C5",
    "slot": 67,
    "name": "Vertebra C5",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      -1.72,
      59.09,
      246.6
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "fifth cervical vertebra"
    ]
  },
  {
    "id": "vertebrae_C6",
    "slot": 68,
    "name": "Vertebra C6",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      -0.35,
      55.99,
      239.48
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "sixth cervical vertebra"
    ]
  },
  {
    "id": "vertebrae_C7",
    "slot": 69,
    "name": "Vertebra C7 (Prominens)",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      0.1,
      43.75,
      231.66
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "seventh cervical vertebra"
    ]
  },
  {
    "id": "vertebrae_T1",
    "slot": 70,
    "name": "Vertebra T1",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      -0.36,
      29.02,
      220.13
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "first thoracic vertebra"
    ]
  },
  {
    "id": "vertebrae_T2",
    "slot": 71,
    "name": "Vertebra T2",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      -0.58,
      17.9,
      203.81
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "second thoracic vertebra"
    ]
  },
  {
    "id": "vertebrae_T3",
    "slot": 72,
    "name": "Vertebra T3",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      -0.3,
      6.38,
      187.33
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "third thoracic vertebra"
    ]
  },
  {
    "id": "vertebrae_T4",
    "slot": 73,
    "name": "Vertebra T4",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      0.37,
      -2.79,
      168.82
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "fourth thoracic vertebra"
    ]
  },
  {
    "id": "vertebrae_T5",
    "slot": 74,
    "name": "Vertebra T5",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      1.6,
      -10.99,
      149.51
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "fifth thoracic vertebra"
    ]
  },
  {
    "id": "vertebrae_T6",
    "slot": 75,
    "name": "Vertebra T6",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      2.26,
      -16.7,
      128.52
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "sixth thoracic vertebra"
    ]
  },
  {
    "id": "vertebrae_T7",
    "slot": 76,
    "name": "Vertebra T7",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      2.6,
      -19.0,
      107.85
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "seventh thoracic vertebra"
    ]
  },
  {
    "id": "vertebrae_T8",
    "slot": 77,
    "name": "Vertebra T8",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      1.83,
      -17.97,
      89.05
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "eighth thoracic vertebra"
    ]
  },
  {
    "id": "vertebrae_T9",
    "slot": 78,
    "name": "Vertebra T9",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      1.68,
      -14.67,
      70.69
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "ninth thoracic vertebra"
    ]
  },
  {
    "id": "vertebrae_T10",
    "slot": 79,
    "name": "Vertebra T10",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      1.25,
      -9.63,
      50.52
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "tenth thoracic vertebra"
    ]
  },
  {
    "id": "vertebrae_T11",
    "slot": 80,
    "name": "Vertebra T11",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      0.84,
      -5.0,
      26.59
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "eleventh thoracic vertebra"
    ]
  },
  {
    "id": "vertebrae_T12",
    "slot": 81,
    "name": "Vertebra T12",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      0.38,
      0.15,
      0.06
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "twelfth thoracic vertebra"
    ]
  },
  {
    "id": "vertebrae_L1",
    "slot": 82,
    "name": "Vertebra L1",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      -0.06,
      6.39,
      -29.36
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "first lumbar vertebra"
    ]
  },
  {
    "id": "vertebrae_L2",
    "slot": 83,
    "name": "Vertebra L2",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      -0.35,
      13.02,
      -56.81
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "second lumbar vertebra"
    ]
  },
  {
    "id": "vertebrae_L3",
    "slot": 84,
    "name": "Vertebra L3",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      -0.49,
      18.48,
      -83.05
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "third lumbar vertebra"
    ]
  },
  {
    "id": "vertebrae_L4",
    "slot": 85,
    "name": "Vertebra L4",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      -0.53,
      23.09,
      -109.59
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "fourth lumbar vertebra"
    ]
  },
  {
    "id": "vertebrae_L5",
    "slot": 86,
    "name": "Vertebra L5",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      -0.1,
      20.69,
      -138.31
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "fifth lumbar vertebra"
    ]
  },
  {
    "id": "vertebrae_S1",
    "slot": 87,
    "name": "Vertebra S1",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      0.25,
      10.41,
      -163.61
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "first sacral vertebra"
    ]
  },
  {
    "id": "autochthon_left",
    "slot": 88,
    "name": "Left Erector Spinae",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      -29.2,
      -28.81,
      8.54
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "left autochthonous back muscle"
    ]
  },
  {
    "id": "autochthon_right",
    "slot": 89,
    "name": "Right Erector Spinae",
    "category": "Spine & Vertebral Column",
    "system": "spine",
    "coords": [
      28.69,
      -28.08,
      7.58
    ],
    "color": 16096779,
    "hex": "#f59e0b",
    "side": "posterior",
    "synonyms": [
      "right autochthonous back muscle"
    ]
  },
  {
    "id": "sternum",
    "slot": 90,
    "name": "Sternum",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      -1.12,
      131.0,
      121.04
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "anterior",
    "synonyms": [
      "breastbone"
    ]
  },
  {
    "id": "costal_cartilages",
    "slot": 91,
    "name": "Costal Cartilages",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      -3.15,
      138.01,
      53.71
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "anterior",
    "synonyms": [
      "rib cartilages"
    ]
  },
  {
    "id": "clavicula_left",
    "slot": 92,
    "name": "Left Clavicle",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      -55.69,
      74.37,
      210.37
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "anterior",
    "synonyms": [
      "left collarbone"
    ]
  },
  {
    "id": "clavicula_right",
    "slot": 93,
    "name": "Right Clavicle",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      57.93,
      72.88,
      210.66
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "anterior",
    "synonyms": [
      "right collarbone"
    ]
  },
  {
    "id": "scapula_left",
    "slot": 94,
    "name": "Left Scapula",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      -117.34,
      -3.35,
      164.92
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "posterior",
    "synonyms": [
      "left shoulder blade"
    ]
  },
  {
    "id": "scapula_right",
    "slot": 95,
    "name": "Right Scapula",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      118.22,
      -0.7,
      163.33
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "posterior",
    "synonyms": [
      "right shoulder blade"
    ]
  },
  {
    "id": "humerus_left",
    "slot": 96,
    "name": "Left Humerus",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      -147.81,
      41.55,
      219.87
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "lateral",
    "synonyms": [
      "left upper arm bone"
    ]
  },
  {
    "id": "humerus_right",
    "slot": 97,
    "name": "Right Humerus",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      144.62,
      44.5,
      214.73
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "lateral",
    "synonyms": [
      "right upper arm bone"
    ]
  },
  {
    "id": "rib_left_1",
    "slot": 98,
    "name": "Rib 1 (Left)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      -55.38,
      66.51,
      199.02
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "anterior",
    "synonyms": [
      "first left rib"
    ]
  },
  {
    "id": "rib_left_2",
    "slot": 99,
    "name": "Rib 2 (Left)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      -73.9,
      51.8,
      190.03
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "anterior",
    "synonyms": [
      "second left rib"
    ]
  },
  {
    "id": "rib_left_3",
    "slot": 100,
    "name": "Rib 3 (Left)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      -84.63,
      48.61,
      168.14
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "anterior",
    "synonyms": [
      "third left rib"
    ]
  },
  {
    "id": "rib_left_4",
    "slot": 101,
    "name": "Rib 4 (Left)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      -91.14,
      54.91,
      140.99
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "anterior",
    "synonyms": [
      "fourth left rib"
    ]
  },
  {
    "id": "rib_left_5",
    "slot": 102,
    "name": "Rib 5 (Left)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      -96.1,
      62.75,
      115.36
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "anterior",
    "synonyms": [
      "fifth left rib"
    ]
  },
  {
    "id": "rib_left_6",
    "slot": 103,
    "name": "Rib 6 (Left)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      -104.85,
      62.72,
      92.5
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "anterior",
    "synonyms": [
      "sixth left rib"
    ]
  },
  {
    "id": "rib_left_7",
    "slot": 104,
    "name": "Rib 7 (Left)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      -113.0,
      50.09,
      73.44
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "anterior",
    "synonyms": [
      "seventh left rib"
    ]
  },
  {
    "id": "rib_left_8",
    "slot": 105,
    "name": "Rib 8 (Left)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      -113.64,
      28.32,
      56.48
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "lateral",
    "synonyms": [
      "eighth left rib"
    ]
  },
  {
    "id": "rib_left_9",
    "slot": 106,
    "name": "Rib 9 (Left)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      -109.24,
      7.2,
      39.05
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "lateral",
    "synonyms": [
      "ninth left rib"
    ]
  },
  {
    "id": "rib_left_10",
    "slot": 107,
    "name": "Rib 10 (Left)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      -100.13,
      -8.91,
      16.7
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "lateral",
    "synonyms": [
      "tenth left rib"
    ]
  },
  {
    "id": "rib_left_11",
    "slot": 108,
    "name": "Rib 11 (Left)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      -82.43,
      -20.48,
      -5.9
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "posterior",
    "synonyms": [
      "eleventh left floating rib"
    ]
  },
  {
    "id": "rib_left_12",
    "slot": 109,
    "name": "Rib 12 (Left)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      -57.7,
      -21.41,
      -22.43
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "posterior",
    "synonyms": [
      "twelfth left floating rib"
    ]
  },
  {
    "id": "rib_right_1",
    "slot": 110,
    "name": "Rib 1 (Right)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      54.42,
      65.62,
      198.19
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "anterior",
    "synonyms": [
      "first right rib"
    ]
  },
  {
    "id": "rib_right_2",
    "slot": 111,
    "name": "Rib 2 (Right)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      72.79,
      52.09,
      188.99
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "anterior",
    "synonyms": [
      "second right rib"
    ]
  },
  {
    "id": "rib_right_3",
    "slot": 112,
    "name": "Rib 3 (Right)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      81.86,
      50.33,
      166.24
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "anterior",
    "synonyms": [
      "third right rib"
    ]
  },
  {
    "id": "rib_right_4",
    "slot": 113,
    "name": "Rib 4 (Right)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      88.59,
      54.3,
      139.34
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "anterior",
    "synonyms": [
      "fourth right rib"
    ]
  },
  {
    "id": "rib_right_5",
    "slot": 114,
    "name": "Rib 5 (Right)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      93.14,
      61.58,
      115.31
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "anterior",
    "synonyms": [
      "fifth right rib"
    ]
  },
  {
    "id": "rib_right_6",
    "slot": 115,
    "name": "Rib 6 (Right)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      101.77,
      62.22,
      94.05
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "anterior",
    "synonyms": [
      "sixth right rib"
    ]
  },
  {
    "id": "rib_right_7",
    "slot": 116,
    "name": "Rib 7 (Right)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      110.66,
      50.79,
      73.58
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "anterior",
    "synonyms": [
      "seventh right rib"
    ]
  },
  {
    "id": "rib_right_8",
    "slot": 117,
    "name": "Rib 8 (Right)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      113.43,
      29.67,
      55.42
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "lateral",
    "synonyms": [
      "eighth right rib"
    ]
  },
  {
    "id": "rib_right_9",
    "slot": 118,
    "name": "Rib 9 (Right)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      109.79,
      7.53,
      37.97
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "lateral",
    "synonyms": [
      "ninth right rib"
    ]
  },
  {
    "id": "rib_right_10",
    "slot": 119,
    "name": "Rib 10 (Right)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      100.38,
      -8.86,
      15.53
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "lateral",
    "synonyms": [
      "tenth right rib"
    ]
  },
  {
    "id": "rib_right_11",
    "slot": 120,
    "name": "Rib 11 (Right)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      83.31,
      -19.97,
      -8.36
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "posterior",
    "synonyms": [
      "eleventh right floating rib"
    ]
  },
  {
    "id": "rib_right_12",
    "slot": 121,
    "name": "Rib 12 (Right)",
    "category": "Ribs & Thoracic Cage",
    "system": "ribs",
    "coords": [
      58.41,
      -20.83,
      -25.56
    ],
    "color": 1357990,
    "hex": "#14b8a6",
    "side": "posterior",
    "synonyms": [
      "twelfth right floating rib"
    ]
  }
];
