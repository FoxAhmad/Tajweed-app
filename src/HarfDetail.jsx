import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import './HarfDetail.css';

const HarfDetail = () => {
  const { harfId } = useParams();
  const navigate = useNavigate();

  const harfDetails = {
    1: { name: 'ا', pronunciation: 'Alif', description: 'The first letter of Arabic alphabet, pronounced as "Alif".' },
    2: { name: 'ب', pronunciation: 'Be', description: 'The second letter, pronounced as "Be".' },
    // Add details for other harfs
  };

  const harf = harfDetails[harfId];

  const handlePracticeClick = () => {
    navigate(`/recording`);
  };

  return (
    <div className="harf-detail">
      <h2>{harf.name} - {harf.pronunciation}</h2>
      <p>{harf.description}</p>
      <audio controls>
        <source src={`audio/${harf.name}.mp3`} type="audio/mp3" />
        Your browser does not support the audio element.
      </audio>
      <button className="practice-btn" onClick={handlePracticeClick}>
        Start Recording
      </button>
    </div>
  );
};

export default HarfDetail;
