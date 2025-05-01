// File: src/HarfSelection.jsx
import React from "react";
import { useNavigate } from "react-router-dom";
//import { FaHome } from "react-icons/fa";
import { Mic } from "lucide-react";
import "./HarfSelection.css"; // Optional if you want to move styles

const haroof = [
  "ا", "ب", "ت", "ث", "ج", "ح", "خ",
  "د", "ذ", "ر", "ز", "س", "ش", "ص",
  "ض", "ط", "ظ", "ع", "غ", "ف", "ق",
  "ك", "ل", "م", "ن", "و", "ه", "ی"
];

const HarfSelection = () => {
  const navigate = useNavigate();

  const playAudio = (index) => {
    const audio = new Audio(`/sounds/arabic_letter_voice_${index + 1}.wav`);
    audio.play();
  };

  const goToHome = () => {
    navigate("/");
  };

  return (
    <div className="harf-page-container">
      <h1 className="page-title">تجوید کے حروف</h1>

      <div className="harf-grid">
        {haroof.map((harf, index) => (
          <div
            key={index}
            className="harf-card"
            onClick={() => playAudio(index)}
          >
            <span className="harf">{harf}</span>
            <span className="practice-icon" onClick={(e) => {
              e.stopPropagation(); // prevent triggering audio
              goToHome();
            }}>
              <Mic />
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default HarfSelection;
