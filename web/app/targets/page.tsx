"use client";

import React, { useState } from "react";
import { Search, Tag, Filter, CheckCircle2 } from "lucide-react";
import { TargetBodyMap } from "@/components/TargetBodyMap";

interface CategoryData {
  name: string;
  description: string;
  targets: { slot: number; name: string; synonyms: string[] }[];
}

const ALL_CATEGORIES: Record<string, CategoryData> = {
  "cranial": {
    "name": "Cranial & Neuroanatomy",
    "description": "Intracranial cerebrum, calvarium, and cranial vault landmarks (Slots 1-2)",
    "targets": [
      {
        "slot": 1,
        "name": "brain",
        "synonyms": [
          "cerebrum",
          "encephalon",
          "cranial cavity",
          "brainstem"
        ]
      },
      {
        "slot": 2,
        "name": "skull",
        "synonyms": [
          "cranium",
          "calvarium",
          "neurocranium"
        ]
      }
    ]
  },
  "thoracic": {
    "name": "Thoracic & Respiratory",
    "description": "Pulmonary lobes, mediastinal structures, and airway conduits (Slots 3-11)",
    "targets": [
      {
        "slot": 3,
        "name": "heart",
        "synonyms": [
          "cardiac organ",
          "cor",
          "myocardium"
        ]
      },
      {
        "slot": 4,
        "name": "trachea",
        "synonyms": [
          "windpipe",
          "respiratory tube"
        ]
      },
      {
        "slot": 5,
        "name": "esophagus",
        "synonyms": [
          "gullet",
          "alimentary canal"
        ]
      },
      {
        "slot": 6,
        "name": "thyroid_gland",
        "synonyms": [
          "thyroid",
          "glandula thyroidea"
        ]
      },
      {
        "slot": 7,
        "name": "lung_upper_lobe_left",
        "synonyms": [
          "LUL",
          "left upper lung"
        ]
      },
      {
        "slot": 8,
        "name": "lung_lower_lobe_left",
        "synonyms": [
          "LLL",
          "left lower lung"
        ]
      },
      {
        "slot": 9,
        "name": "lung_upper_lobe_right",
        "synonyms": [
          "RUL",
          "right upper lung"
        ]
      },
      {
        "slot": 10,
        "name": "lung_middle_lobe_right",
        "synonyms": [
          "RML",
          "right middle lung"
        ]
      },
      {
        "slot": 11,
        "name": "lung_lower_lobe_right",
        "synonyms": [
          "RLL",
          "right lower lung"
        ]
      }
    ]
  },
  "vascular": {
    "name": "Cardiovascular & Major Vessels",
    "description": "Great arteries, central veins, and systemic vascular tree (Slots 12-28)",
    "targets": [
      {
        "slot": 12,
        "name": "aorta",
        "synonyms": [
          "thoracic aorta",
          "abdominal aorta"
        ]
      },
      {
        "slot": 13,
        "name": "superior_vena_cava",
        "synonyms": [
          "SVC",
          "precava"
        ]
      },
      {
        "slot": 14,
        "name": "inferior_vena_cava",
        "synonyms": [
          "IVC",
          "postcava"
        ]
      },
      {
        "slot": 15,
        "name": "pulmonary_vein",
        "synonyms": [
          "venae pulmonales"
        ]
      },
      {
        "slot": 16,
        "name": "brachiocephalic_trunk",
        "synonyms": [
          "innominate artery"
        ]
      },
      {
        "slot": 17,
        "name": "subclavian_artery_right",
        "synonyms": [
          "right subclavian"
        ]
      },
      {
        "slot": 18,
        "name": "subclavian_artery_left",
        "synonyms": [
          "left subclavian"
        ]
      },
      {
        "slot": 19,
        "name": "common_carotid_artery_right",
        "synonyms": [
          "right carotid"
        ]
      },
      {
        "slot": 20,
        "name": "common_carotid_artery_left",
        "synonyms": [
          "left carotid"
        ]
      },
      {
        "slot": 21,
        "name": "brachiocephalic_vein_left",
        "synonyms": [
          "left innominate vein"
        ]
      },
      {
        "slot": 22,
        "name": "brachiocephalic_vein_right",
        "synonyms": [
          "right innominate vein"
        ]
      },
      {
        "slot": 23,
        "name": "atrial_appendage_left",
        "synonyms": [
          "left auricle"
        ]
      },
      {
        "slot": 24,
        "name": "portal_vein_and_splenic_vein",
        "synonyms": [
          "portal venous confluence"
        ]
      },
      {
        "slot": 25,
        "name": "iliac_artery_left",
        "synonyms": [
          "left common iliac artery"
        ]
      },
      {
        "slot": 26,
        "name": "iliac_artery_right",
        "synonyms": [
          "right common iliac artery"
        ]
      },
      {
        "slot": 27,
        "name": "iliac_vena_left",
        "synonyms": [
          "left common iliac vein"
        ]
      },
      {
        "slot": 28,
        "name": "iliac_vena_right",
        "synonyms": [
          "right common iliac vein"
        ]
      }
    ]
  },
  "abdominal": {
    "name": "Abdominal & Digestive",
    "description": "Visceral digestive organs, hepatic, splenic, renal, and biliary glands (Slots 29-42)",
    "targets": [
      {
        "slot": 29,
        "name": "liver",
        "synonyms": [
          "hepatic parenchyma",
          "hepar"
        ]
      },
      {
        "slot": 30,
        "name": "spleen",
        "synonyms": [
          "lien",
          "splenic tissue"
        ]
      },
      {
        "slot": 31,
        "name": "pancreas",
        "synonyms": [
          "pancreatic gland",
          "pancreatic body"
        ]
      },
      {
        "slot": 32,
        "name": "gallbladder",
        "synonyms": [
          "cholecyst",
          "biliary vesicle"
        ]
      },
      {
        "slot": 33,
        "name": "stomach",
        "synonyms": [
          "gaster",
          "ventriculus"
        ]
      },
      {
        "slot": 34,
        "name": "duodenum",
        "synonyms": [
          "proximal small intestine"
        ]
      },
      {
        "slot": 35,
        "name": "small_bowel",
        "synonyms": [
          "jejunum",
          "ileum",
          "small intestine"
        ]
      },
      {
        "slot": 36,
        "name": "colon",
        "synonyms": [
          "large bowel",
          "large intestine"
        ]
      },
      {
        "slot": 37,
        "name": "kidney_right",
        "synonyms": [
          "right renal gland",
          "ren dexter"
        ]
      },
      {
        "slot": 38,
        "name": "kidney_left",
        "synonyms": [
          "left renal gland",
          "ren sinister"
        ]
      },
      {
        "slot": 39,
        "name": "adrenal_gland_right",
        "synonyms": [
          "right suprarenal gland"
        ]
      },
      {
        "slot": 40,
        "name": "adrenal_gland_left",
        "synonyms": [
          "left suprarenal gland"
        ]
      },
      {
        "slot": 41,
        "name": "kidney_cyst_left",
        "synonyms": [
          "left renal cyst"
        ]
      },
      {
        "slot": 42,
        "name": "kidney_cyst_right",
        "synonyms": [
          "right renal cyst"
        ]
      }
    ]
  },
  "pelvis": {
    "name": "Pelvis, Reproductive & Locomotion",
    "description": "Bladder, female Uterus & Ovaries, male Prostate, pelvic bones, and hip musculature (Slots 43-61)",
    "targets": [
      {
        "slot": 43,
        "name": "urinary_bladder",
        "synonyms": [
          "vesica urinaria",
          "bladder"
        ]
      },
      {
        "slot": 44,
        "name": "prostate",
        "synonyms": [
          "prostatic gland",
          "male reproductive"
        ]
      },
      {
        "slot": 45,
        "name": "uterus",
        "synonyms": [
          "womb",
          "uterine body",
          "myometrium",
          "female pelvis"
        ]
      },
      {
        "slot": 46,
        "name": "ovary_left",
        "synonyms": [
          "left adnexa",
          "ovarium sinistrum",
          "female gonad"
        ]
      },
      {
        "slot": 47,
        "name": "ovary_right",
        "synonyms": [
          "right adnexa",
          "ovarium dextrum",
          "female gonad"
        ]
      },
      {
        "slot": 48,
        "name": "vagina",
        "synonyms": [
          "vaginal canal",
          "female birth canal",
          "colpos"
        ]
      },
      {
        "slot": 49,
        "name": "sacrum",
        "synonyms": [
          "sacral bone",
          "sacral vertebra"
        ]
      },
      {
        "slot": 50,
        "name": "hip_left",
        "synonyms": [
          "left ilium",
          "left pelvic bone"
        ]
      },
      {
        "slot": 51,
        "name": "hip_right",
        "synonyms": [
          "right ilium",
          "right pelvic bone"
        ]
      },
      {
        "slot": 52,
        "name": "femur_left",
        "synonyms": [
          "left thigh bone"
        ]
      },
      {
        "slot": 53,
        "name": "femur_right",
        "synonyms": [
          "right thigh bone"
        ]
      },
      {
        "slot": 54,
        "name": "gluteus_maximus_left",
        "synonyms": [
          "left upper buttock muscle"
        ]
      },
      {
        "slot": 55,
        "name": "gluteus_maximus_right",
        "synonyms": [
          "right upper buttock muscle"
        ]
      },
      {
        "slot": 56,
        "name": "gluteus_medius_left",
        "synonyms": [
          "left middle buttock muscle"
        ]
      },
      {
        "slot": 57,
        "name": "gluteus_medius_right",
        "synonyms": [
          "right middle buttock muscle"
        ]
      },
      {
        "slot": 58,
        "name": "gluteus_minimus_left",
        "synonyms": [
          "left deep buttock muscle"
        ]
      },
      {
        "slot": 59,
        "name": "gluteus_minimus_right",
        "synonyms": [
          "right deep buttock muscle"
        ]
      },
      {
        "slot": 60,
        "name": "iliopsoas_left",
        "synonyms": [
          "left psoas hip flexor"
        ]
      },
      {
        "slot": 61,
        "name": "iliopsoas_right",
        "synonyms": [
          "right psoas hip flexor"
        ]
      }
    ]
  },
  "spine": {
    "name": "Spine & Vertebral Column",
    "description": "Cervical, Thoracic, Lumbar vertebrae, spinal cord, and erector spinae (Slots 62-89)",
    "targets": [
      {
        "slot": 62,
        "name": "spinal_cord",
        "synonyms": [
          "medulla spinalis",
          "spinal canal"
        ]
      },
      {
        "slot": 63,
        "name": "vertebrae_C1",
        "synonyms": [
          "atlas vertebra"
        ]
      },
      {
        "slot": 64,
        "name": "vertebrae_C2",
        "synonyms": [
          "axis vertebra"
        ]
      },
      {
        "slot": 65,
        "name": "vertebrae_C3",
        "synonyms": [
          "third cervical vertebra"
        ]
      },
      {
        "slot": 66,
        "name": "vertebrae_C4",
        "synonyms": [
          "fourth cervical vertebra"
        ]
      },
      {
        "slot": 67,
        "name": "vertebrae_C5",
        "synonyms": [
          "fifth cervical vertebra"
        ]
      },
      {
        "slot": 68,
        "name": "vertebrae_C6",
        "synonyms": [
          "sixth cervical vertebra"
        ]
      },
      {
        "slot": 69,
        "name": "vertebrae_C7",
        "synonyms": [
          "seventh cervical vertebra"
        ]
      },
      {
        "slot": 70,
        "name": "vertebrae_T1",
        "synonyms": [
          "first thoracic vertebra"
        ]
      },
      {
        "slot": 71,
        "name": "vertebrae_T2",
        "synonyms": [
          "second thoracic vertebra"
        ]
      },
      {
        "slot": 72,
        "name": "vertebrae_T3",
        "synonyms": [
          "third thoracic vertebra"
        ]
      },
      {
        "slot": 73,
        "name": "vertebrae_T4",
        "synonyms": [
          "fourth thoracic vertebra"
        ]
      },
      {
        "slot": 74,
        "name": "vertebrae_T5",
        "synonyms": [
          "fifth thoracic vertebra"
        ]
      },
      {
        "slot": 75,
        "name": "vertebrae_T6",
        "synonyms": [
          "sixth thoracic vertebra"
        ]
      },
      {
        "slot": 76,
        "name": "vertebrae_T7",
        "synonyms": [
          "seventh thoracic vertebra"
        ]
      },
      {
        "slot": 77,
        "name": "vertebrae_T8",
        "synonyms": [
          "eighth thoracic vertebra"
        ]
      },
      {
        "slot": 78,
        "name": "vertebrae_T9",
        "synonyms": [
          "ninth thoracic vertebra"
        ]
      },
      {
        "slot": 79,
        "name": "vertebrae_T10",
        "synonyms": [
          "tenth thoracic vertebra"
        ]
      },
      {
        "slot": 80,
        "name": "vertebrae_T11",
        "synonyms": [
          "eleventh thoracic vertebra"
        ]
      },
      {
        "slot": 81,
        "name": "vertebrae_T12",
        "synonyms": [
          "twelfth thoracic vertebra"
        ]
      },
      {
        "slot": 82,
        "name": "vertebrae_L1",
        "synonyms": [
          "first lumbar vertebra"
        ]
      },
      {
        "slot": 83,
        "name": "vertebrae_L2",
        "synonyms": [
          "second lumbar vertebra"
        ]
      },
      {
        "slot": 84,
        "name": "vertebrae_L3",
        "synonyms": [
          "third lumbar vertebra"
        ]
      },
      {
        "slot": 85,
        "name": "vertebrae_L4",
        "synonyms": [
          "fourth lumbar vertebra"
        ]
      },
      {
        "slot": 86,
        "name": "vertebrae_L5",
        "synonyms": [
          "fifth lumbar vertebra"
        ]
      },
      {
        "slot": 87,
        "name": "vertebrae_S1",
        "synonyms": [
          "first sacral vertebra"
        ]
      },
      {
        "slot": 88,
        "name": "autochthon_left",
        "synonyms": [
          "left autochthonous back muscle"
        ]
      },
      {
        "slot": 89,
        "name": "autochthon_right",
        "synonyms": [
          "right autochthonous back muscle"
        ]
      }
    ]
  },
  "ribs": {
    "name": "Ribs & Thoracic Cage",
    "description": "Bilateral costal cage (Ribs 1-12 Left & Right), sternum, clavicles, scapulae, humerus (Slots 90-121)",
    "targets": [
      {
        "slot": 90,
        "name": "sternum",
        "synonyms": [
          "breastbone"
        ]
      },
      {
        "slot": 91,
        "name": "costal_cartilages",
        "synonyms": [
          "rib cartilages"
        ]
      },
      {
        "slot": 92,
        "name": "clavicula_left",
        "synonyms": [
          "left collarbone"
        ]
      },
      {
        "slot": 93,
        "name": "clavicula_right",
        "synonyms": [
          "right collarbone"
        ]
      },
      {
        "slot": 94,
        "name": "scapula_left",
        "synonyms": [
          "left shoulder blade"
        ]
      },
      {
        "slot": 95,
        "name": "scapula_right",
        "synonyms": [
          "right shoulder blade"
        ]
      },
      {
        "slot": 96,
        "name": "humerus_left",
        "synonyms": [
          "left upper arm bone"
        ]
      },
      {
        "slot": 97,
        "name": "humerus_right",
        "synonyms": [
          "right upper arm bone"
        ]
      },
      {
        "slot": 98,
        "name": "rib_left_1",
        "synonyms": [
          "first left rib"
        ]
      },
      {
        "slot": 99,
        "name": "rib_left_2",
        "synonyms": [
          "second left rib"
        ]
      },
      {
        "slot": 100,
        "name": "rib_left_3",
        "synonyms": [
          "third left rib"
        ]
      },
      {
        "slot": 101,
        "name": "rib_left_4",
        "synonyms": [
          "fourth left rib"
        ]
      },
      {
        "slot": 102,
        "name": "rib_left_5",
        "synonyms": [
          "fifth left rib"
        ]
      },
      {
        "slot": 103,
        "name": "rib_left_6",
        "synonyms": [
          "sixth left rib"
        ]
      },
      {
        "slot": 104,
        "name": "rib_left_7",
        "synonyms": [
          "seventh left rib"
        ]
      },
      {
        "slot": 105,
        "name": "rib_left_8",
        "synonyms": [
          "eighth left rib"
        ]
      },
      {
        "slot": 106,
        "name": "rib_left_9",
        "synonyms": [
          "ninth left rib"
        ]
      },
      {
        "slot": 107,
        "name": "rib_left_10",
        "synonyms": [
          "tenth left rib"
        ]
      },
      {
        "slot": 108,
        "name": "rib_left_11",
        "synonyms": [
          "eleventh left floating rib"
        ]
      },
      {
        "slot": 109,
        "name": "rib_left_12",
        "synonyms": [
          "twelfth left floating rib"
        ]
      },
      {
        "slot": 110,
        "name": "rib_right_1",
        "synonyms": [
          "first right rib"
        ]
      },
      {
        "slot": 111,
        "name": "rib_right_2",
        "synonyms": [
          "second right rib"
        ]
      },
      {
        "slot": 112,
        "name": "rib_right_3",
        "synonyms": [
          "third right rib"
        ]
      },
      {
        "slot": 113,
        "name": "rib_right_4",
        "synonyms": [
          "fourth right rib"
        ]
      },
      {
        "slot": 114,
        "name": "rib_right_5",
        "synonyms": [
          "fifth right rib"
        ]
      },
      {
        "slot": 115,
        "name": "rib_right_6",
        "synonyms": [
          "sixth right rib"
        ]
      },
      {
        "slot": 116,
        "name": "rib_right_7",
        "synonyms": [
          "seventh right rib"
        ]
      },
      {
        "slot": 117,
        "name": "rib_right_8",
        "synonyms": [
          "eighth right rib"
        ]
      },
      {
        "slot": 118,
        "name": "rib_right_9",
        "synonyms": [
          "ninth right rib"
        ]
      },
      {
        "slot": 119,
        "name": "rib_right_10",
        "synonyms": [
          "tenth right rib"
        ]
      },
      {
        "slot": 120,
        "name": "rib_right_11",
        "synonyms": [
          "eleventh right floating rib"
        ]
      },
      {
        "slot": 121,
        "name": "rib_right_12",
        "synonyms": [
          "twelfth right floating rib"
        ]
      }
    ]
  }
};

export default function TargetsPage() {
  const [query, setQuery] = useState("");
  const [selectedCat, setSelectedCat] = useState<string>("all");
  const [selectedPin, setSelectedPin] = useState<string | null>(null);


  const totalTargetsCount = Object.values(ALL_CATEGORIES).reduce(
    (acc, cat) => acc + cat.targets.length,
    0
  );

  const handlePinSelect = (organName: string) => {
    setSelectedPin(organName);
    setTimeout(() => {
      const el = document.getElementById(`target-card-${organName}`);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }, 80);
  };

  const handleSystemFilterFrom3D = (sys: string | null) => {
    if (!sys || sys === "all") {
      setSelectedCat("all");
    } else {
      setSelectedCat(sys);
    }
  };

  const filteredCategories = Object.entries(ALL_CATEGORIES).filter(([key]) => {
    if (selectedCat !== "all" && selectedCat !== key) return false;
    return true;
  });

  return (
    <div className="max-w-[1520px] mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col gap-5">
      {/* Header Banner */}
      <div className="bg-white p-4 sm:p-5 rounded-2xl border border-border shadow-xs flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-semibold px-2.5 py-0.5 rounded-full bg-primary/10 text-primary-dark border border-primary/20">
              121 Canonical Landmarks Active
            </span>
            <span className="text-xs font-mono text-slate-500 hidden md:inline">
              Cranial (1-2) • Thorax (3-11) • Vascular (12-28) • Abdomen (29-42) • Pelvis (43-61) • Spine (62-89) • Ribs (90-121)
            </span>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold text-text-main mt-1.5">121 Anatomical Target Catalog</h1>
          <p className="text-xs text-text-muted mt-0.5 max-w-3xl leading-relaxed">
            Interactive 3D body map and systematic clinical directory of all 121 internal anatomical landmarks. Click any organ on the right to inspect its 3D spatial position on the doll, or drag to orbit 360°.
          </p>
        </div>
      </div>

      {/* Side-by-Side Dashboard Layout: 3D Doll on Left | Organ Catalog on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left Column: 3D Body Map Doll (lg:col-span-7) */}
        <div className="lg:col-span-7 flex flex-col gap-3 lg:sticky lg:top-20">
          <TargetBodyMap
            onSelectTarget={handlePinSelect}
            selectedTarget={selectedPin}
            activeSystemFilter={selectedCat === "all" ? null : selectedCat}
            onSelectSystemFilter={handleSystemFilterFrom3D}
          />
        </div>

        {/* Right Column: 121 Target Organs Directory (lg:col-span-5) */}
        <div className="lg:col-span-5 flex flex-col gap-3 bg-white p-4 sm:p-5 rounded-3xl border border-border shadow-xs h-[560px] sm:h-[640px] lg:h-[720px] lg:sticky lg:top-20">
          {/* Search Bar */}
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
            <input
              type="text"
              placeholder="Search 121 targets or slot #..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full pl-9 pr-8 py-2 text-xs rounded-xl border border-border bg-slate-50 focus:bg-white focus:outline-none focus:ring-1 focus:ring-primary"
            />
            {query && (
              <button
                onClick={() => setQuery("")}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-xs text-slate-400 hover:text-slate-600"
              >
                ✕
              </button>
            )}
          </div>

          {/* Region Filter Chips */}
          <div className="flex items-center gap-1.5 overflow-x-auto no-scrollbar pb-1 text-xs shrink-0">
            <button
              onClick={() => setSelectedCat("all")}
              className={`px-2.5 py-1 rounded-xl text-[11px] font-medium border transition-colors shrink-0 ${
                selectedCat === "all"
                  ? "bg-slate-900 text-white font-semibold shadow-xs"
                  : "bg-slate-50 hover:bg-slate-100 text-slate-600 border-slate-200"
              }`}
            >
              All 121
            </button>
            {Object.entries(ALL_CATEGORIES).map(([key, cat]) => (
              <button
                key={key}
                onClick={() => setSelectedCat(key)}
                className={`px-2.5 py-1 rounded-xl text-[11px] font-medium border transition-colors whitespace-nowrap shrink-0 ${
                  selectedCat === key
                    ? "bg-slate-900 text-white font-semibold shadow-xs"
                    : "bg-slate-50 hover:bg-slate-100 text-slate-600 border-slate-200"
                }`}
              >
                {cat.name.split(" ")[0]} ({cat.targets.length})
              </button>
            ))}
          </div>

          {/* Active Pin Focus HUD Card (if selected) */}
          {selectedPin && (
            <div className="bg-amber-50/80 p-3 rounded-2xl border border-amber-300 flex items-center justify-between gap-3 shrink-0 animate-fade-in">
              <div className="flex items-center gap-2 min-w-0">
                <span className="w-2.5 h-2.5 rounded-full bg-amber-500 animate-ping shrink-0" />
                <div className="min-w-0">
                  <div className="text-[10px] text-amber-800 uppercase font-mono font-bold">Focused 3D Landmark</div>
                  <div className="text-sm font-bold text-slate-900 capitalize truncate">
                    {selectedPin.replace(/_/g, " ")}
                  </div>
                </div>
              </div>
              <button
                onClick={() => setSelectedPin(null)}
                className="text-[11px] font-semibold px-2.5 py-1 rounded-lg bg-white hover:bg-amber-100 text-amber-900 border border-amber-300 transition-colors shrink-0"
              >
                Clear Focus
              </button>
            </div>
          )}

          {/* Scrollable Target Cards List */}
          <div className="flex-1 overflow-y-auto pr-1 flex flex-col gap-4">
            {filteredCategories.map(([key, cat]) => {
              const filteredTargets = cat.targets.filter((t) => {
                if (!query.trim()) return true;
                const q = query.toLowerCase().trim();
                if (t.name.toLowerCase().includes(q)) return true;
                if (String(t.slot) === q) return true;
                if (t.synonyms.some((s) => s.toLowerCase().includes(q))) return true;
                return false;
              });

              if (filteredTargets.length === 0) return null;

              return (
                <div key={key} className="flex flex-col gap-2">
                  <div className="flex items-center justify-between border-b border-border/80 pb-1.5 pt-1">
                    <h3 className="text-xs font-bold text-text-main flex items-center gap-1.5">
                      <span>{cat.name}</span>
                      <span className="text-[10px] font-mono font-normal text-text-muted px-1.5 py-0.2 rounded bg-slate-100 border border-slate-200">
                        {filteredTargets.length}
                      </span>
                    </h3>
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    {filteredTargets.map((t) => {
                      const isSelected = selectedPin === t.name;
                      return (
                        <div
                          key={t.name}
                          id={`target-card-${t.name}`}
                          onClick={() => handlePinSelect(t.name)}
                          className={`p-2.5 rounded-xl border transition-all cursor-pointer flex flex-col gap-1 ${
                            isSelected
                              ? "border-amber-400 bg-amber-50/60 ring-2 ring-amber-400/40 shadow-xs"
                              : "border-slate-200 hover:border-primary/40 bg-slate-50/50 hover:bg-slate-50"
                          }`}
                        >
                          <div className="flex items-center justify-between gap-1">
                            <span className="text-xs font-bold text-text-main capitalize truncate">
                              {t.name.replace(/_/g, " ")}
                            </span>
                            <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-white border border-slate-200 font-bold text-primary-dark shrink-0">
                              #{t.slot}
                            </span>
                          </div>
                          {t.synonyms.length > 0 && (
                            <span className="text-[10px] text-text-muted truncate">
                              {t.synonyms[0]}
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}


