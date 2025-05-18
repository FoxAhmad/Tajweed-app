// File: src/HarfSelection.jsx (Updated)
import React from "react";
import { useNavigate } from "react-router-dom";
import { Mic, Volume2 } from "lucide-react";
import "./HarfSelection.css";

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

  const goToHarfDetail = (index) => {
    navigate(`/harf/${index}`);
  };

  const goToHome = () => {
    navigate("/");
  };

  return (
    <div className="harf-selection-container">
      <div className="harf-selection-header">
        
        <h1 className="page-title">تجوید کے حروف</h1>
        <p className="page-subtitle">Select a letter to practice its pronunciation</p>
      </div>

      <div className="harf-grid">
        {haroof.map((harf, index) => (
          <div key={index} className="harf-card">
            <span className="harf">{harf}</span>
            <div className="harf-actions">
              <button 
                className="action-button play-button"
                onClick={() => playAudio(index)}
                aria-label={`Play ${harf} sound`}
              >
                <Volume2 size={18} />Play
              </button>
              <button 
                className="action-button practice-button"
                onClick={() => goToHarfDetail(index)}
                aria-label={`Practice ${harf} pronunciation`}
              >
                <Mic size={18} />Practice
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default HarfSelection;