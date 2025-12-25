
import React, { useState, useEffect, useMemo, useRef } from 'react';
import ReactDOM from 'react-dom/client';
import { GoogleGenAI, Type, GenerateContentResponse } from "@google/genai";
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area 
} from 'recharts';

// --- TYPES & INTERFACES ---

type Severity = 'Critical' | 'Elevated' | 'Normal';

interface VitalsRecord {
  id: string;
  parameter: string;
  reading: string;
  unit: string;
  severity: Severity;
  timestamp: number;
}

interface MedicalHistoryEntry {
  id: string;
  condition: string;
  status: string;
  date: string;
}

interface Medication {
  name: string;
  dosage: string;
  frequency: string;
}

interface ClinicalAssessment {
  summary: string;
  risks: string[];
  recommendations: string[];
  nextSteps: string[];
}

interface PatientProfile {
  name: string;
  age?: number;
  history: MedicalHistoryEntry[];
  vitals: VitalsRecord[];
  medications: Medication[];
  assessment?: ClinicalAssessment;
}

// --- AI SERVICES ---

const getAI = () => new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

const parseClinicalDocument = async (base64Data: string, mimeType: string) => {
  const ai = getAI();
  const prompt = `Analyze this medical document. Extract:
  1. Patient name and age.
  2. Clinical Vitals (BP, Glucose, Cholesterol, Sodium, etc.) with readings and units.
  3. Medical History (Past conditions or diagnoses).
  4. Current Medications mentioned.
  
  Assign a severity to each vital: 'Critical', 'Elevated', or 'Normal' based on standard clinical thresholds.`;

  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: {
      parts: [
        { inlineData: { data: base64Data, mimeType } },
        { text: prompt }
      ]
    },
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          name: { type: Type.STRING },
          age: { type: Type.INTEGER },
          vitals: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                parameter: { type: Type.STRING },
                reading: { type: Type.STRING },
                unit: { type: Type.STRING },
                severity: { type: Type.STRING, enum: ['Critical', 'Elevated', 'Normal'] }
              }
            }
          },
          history: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                condition: { type: Type.STRING },
                status: { type: Type.STRING },
                date: { type: Type.STRING }
              }
            }
          },
          medications: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                name: { type: Type.STRING },
                dosage: { type: Type.STRING },
                frequency: { type: Type.STRING }
              }
            }
          }
        }
      }
    }
  });

  try {
    return JSON.parse(response.text || '{}');
  } catch (e) {
    console.error("Parse Error", e);
    return null;
  }
};

const generateHealthAssessment = async (profile: PatientProfile): Promise<ClinicalAssessment | null> => {
  const ai = getAI();
  const prompt = `Review the following patient data and provide a professional clinical summary, risk assessment, and personalized recommendations.
  Patient Profile: ${JSON.stringify(profile)}
  
  Return the assessment in a structured format.`;

  const response = await ai.models.generateContent({
    model: 'gemini-3-pro-preview',
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          summary: { type: Type.STRING },
          risks: { type: Type.ARRAY, items: { type: Type.STRING } },
          recommendations: { type: Type.ARRAY, items: { type: Type.STRING } },
          nextSteps: { type: Type.ARRAY, items: { type: Type.STRING } }
        },
        required: ["summary", "risks", "recommendations", "nextSteps"]
      }
    }
  });

  try {
    return JSON.parse(response.text || '{}');
  } catch (e) {
    console.error("Assessment error", e);
    return null;
  }
};

const getClinicalChatResponse = async (query: string, profile: PatientProfile, history: { role: string, text: string }[]) => {
  const ai = getAI();
  const chat = ai.chats.create({
    model: 'gemini-3-pro-preview',
    config: {
      systemInstruction: `You are MedAId, a highly accurate clinical assistant. You have access to the patient's full records: ${JSON.stringify(profile)}. Answer questions concisely and professionally. If a patient asks about symptoms that sound like an emergency, advise them to call emergency services immediately. Always remind the user that you are an AI assistant and they should consult a real medical professional for diagnosis.`
    }
  });
  
  // Send previous history for context if needed, or just the latest query with full profile context
  const response = await chat.sendMessage({ message: query });
  return response.text;
};

const searchMedicalFacilities = async (query: string, lat?: number, lng?: number) => {
  const ai = getAI();
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: `Find high-quality ${query} near ${lat && lng ? `${lat}, ${lng}` : 'my location'}. Provide a brief description and then specific locations.`,
    config: {
      tools: [{ googleMaps: {} }],
      toolConfig: {
        retrievalConfig: {
          latLng: lat && lng ? { latitude: lat, longitude: lng } : undefined
        }
      }
    },
  });

  const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
  const links = groundingChunks
    .filter((c: any) => c.maps)
    .map((c: any) => ({ title: c.maps.title, uri: c.maps.uri }));

  return { text: response.text, links };
};

// --- COMPONENTS ---

const Sidebar: React.FC<{ activeTab: string, setActiveTab: (t: string) => void }> = ({ activeTab, setActiveTab }) => {
  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: 'M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z' },
    { id: 'assessment', label: 'Clinical Insight', icon: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01' },
    { id: 'history', label: 'Medical History', icon: 'M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z' },
    { id: 'chat', label: 'AI Assistant', icon: 'M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z' },
    { id: 'maps', label: 'Care Finder', icon: 'M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z' },
  ];

  return (
    <div className="w-64 bg-slate-900 h-screen fixed left-0 top-0 text-slate-300 p-6 flex flex-col border-r border-slate-800 z-40">
      <div className="flex items-center gap-3 mb-10 group cursor-default">
        <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center font-black text-white text-xl shadow-lg shadow-indigo-900/20 group-hover:scale-110 transition-transform">M</div>
        <h1 className="text-2xl font-bold text-white tracking-tight">MedAId</h1>
      </div>
      <nav className="space-y-2 flex-1">
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-all ${activeTab === t.id ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-900/40' : 'hover:bg-slate-800 hover:text-white'}`}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d={t.icon} />
            </svg>
            {t.label}
          </button>
        ))}
      </nav>
      <div className="mt-auto pt-6 border-t border-slate-800 text-[10px] uppercase tracking-widest text-slate-500 font-bold">
        Certified Clinical AI v2.0
      </div>
    </div>
  );
};

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [profile, setProfile] = useState<PatientProfile>({
    name: 'Unregistered Patient',
    history: [],
    vitals: [],
    medications: []
  });
  const [loading, setLoading] = useState(false);
  const [chatLog, setChatLog] = useState<{ role: string, text: string }[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [mapsResults, setMapsResults] = useState<{ text: string, links: { title: string, uri: string }[] } | null>(null);

  // Persistence
  useEffect(() => {
    const saved = localStorage.getItem('medaid_v2_profile');
    if (saved) setProfile(JSON.parse(saved));
  }, []);

  useEffect(() => {
    localStorage.setItem('medaid_v2_profile', JSON.stringify(profile));
  }, [profile]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);

    const reader = new FileReader();
    reader.onloadend = async () => {
      const base64 = (reader.result as string).split(',')[1];
      const result = await parseClinicalDocument(base64, file.type);
      if (result) {
        setProfile(prev => ({
          ...prev,
          name: result.name || prev.name,
          age: result.age || prev.age,
          vitals: [...prev.vitals, ...(result.vitals || []).map((v: any) => ({ ...v, id: Math.random().toString(), timestamp: Date.now() }))],
          history: [...prev.history, ...(result.history || []).map((h: any) => ({ ...h, id: Math.random().toString() }))],
          medications: result.medications || prev.medications
        }));
      }
      setLoading(false);
    };
    reader.readAsDataURL(file);
  };

  const handleAssessmentGeneration = async () => {
    setLoading(true);
    const assessment = await generateHealthAssessment(profile);
    if (assessment) {
      setProfile(p => ({ ...p, assessment }));
      setActiveTab('assessment');
    }
    setLoading(false);
  };

  const handleChat = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;
    const userMsg = chatInput;
    setChatInput('');
    setChatLog(prev => [...prev, { role: 'user', text: userMsg }]);
    
    setLoading(true);
    const response = await getClinicalChatResponse(userMsg, profile, chatLog);
    setChatLog(prev => [...prev, { role: 'ai', text: response || 'Unable to process query at this time.' }]);
    setLoading(false);
  };

  const handleMapsSearch = async (query: string) => {
    setLoading(true);
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const res = await searchMedicalFacilities(query, pos.coords.latitude, pos.coords.longitude);
        setMapsResults(res);
        setLoading(false);
      },
      async () => {
        const res = await searchMedicalFacilities(query);
        setMapsResults(res);
        setLoading(false);
      }
    );
  };

  const vitalsChartData = useMemo(() => {
    const sorted = [...profile.vitals].sort((a, b) => a.timestamp - b.timestamp);
    return sorted.slice(-12).map(v => ({
      name: new Date(v.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' }),
      value: parseFloat(v.reading) || 0,
      param: v.parameter
    }));
  }, [profile.vitals]);

  const healthScore = useMemo(() => {
    if (profile.vitals.length === 0) return 100;
    const criticals = profile.vitals.filter(v => v.severity === 'Critical').length;
    const elevateds = profile.vitals.filter(v => v.severity === 'Elevated').length;
    return Math.max(0, 100 - (criticals * 15) - (elevateds * 5));
  }, [profile.vitals]);

  return (
    <div className="flex bg-slate-50 min-h-screen font-sans selection:bg-indigo-100 selection:text-indigo-700">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <main className="flex-1 ml-64 p-8 lg:p-12">
        <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 mb-12">
          <div>
            <h2 className="text-4xl font-black text-slate-900 tracking-tight">{profile.name}</h2>
            <div className="flex items-center gap-3 mt-1">
               <span className="flex items-center gap-1.5 text-slate-500 font-medium bg-white px-3 py-1 rounded-full border border-slate-200 text-sm">
                 <div className={`w-2 h-2 rounded-full ${healthScore > 80 ? 'bg-emerald-500' : healthScore > 50 ? 'bg-amber-500' : 'bg-rose-500'}`}></div>
                 {healthScore > 80 ? 'Optimal Status' : healthScore > 50 ? 'Action Recommended' : 'Attention Required'}
               </span>
               <span className="text-slate-400 font-medium">•</span>
               <span className="text-slate-500 font-medium">{profile.age ? `${profile.age} Years Old` : 'Age Not Provided'}</span>
            </div>
          </div>
          <div className="flex gap-3">
            <label className="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-3 rounded-2xl font-bold cursor-pointer transition-all shadow-xl shadow-indigo-200 flex items-center gap-2 text-sm active:scale-95">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"/></svg>
              Upload Clinical Data
              <input type="file" className="hidden" onChange={handleFileUpload} accept="image/*,application/pdf" />
            </label>
            <button onClick={() => {localStorage.clear(); window.location.reload();}} className="w-12 h-12 flex items-center justify-center bg-white border border-slate-200 rounded-2xl text-slate-400 hover:text-rose-500 hover:border-rose-200 transition-all active:scale-95">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
            </button>
          </div>
        </header>

        {loading && (
          <div className="fixed top-0 left-0 w-full h-1 bg-indigo-600 animate-pulse z-50"></div>
        )}

        {activeTab === 'dashboard' && (
          <div className="space-y-8 animate-fade-in">
            {/* Quick Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="bg-white p-7 rounded-[2rem] shadow-sm border border-slate-200">
                <p className="text-slate-400 text-[10px] font-black uppercase tracking-widest mb-3">Overall Health</p>
                <div className="flex items-end gap-2">
                  <span className={`text-5xl font-black ${healthScore > 80 ? 'text-indigo-600' : 'text-amber-600'}`}>{healthScore}</span>
                  <span className="text-slate-300 font-bold mb-1.5">/100</span>
                </div>
              </div>
              <div className="bg-white p-7 rounded-[2rem] shadow-sm border border-slate-200">
                <p className="text-slate-400 text-[10px] font-black uppercase tracking-widest mb-3">Medications</p>
                <div className="flex items-center gap-3">
                  <span className="text-5xl font-black text-slate-900">{profile.medications.length}</span>
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-400"></div>
                </div>
              </div>
              <div className="bg-white p-7 rounded-[2rem] shadow-sm border border-slate-200">
                <p className="text-slate-400 text-[10px] font-black uppercase tracking-widest mb-3">Vitals Logged</p>
                <span className="text-5xl font-black text-slate-900">{profile.vitals.length}</span>
              </div>
              <div className="bg-indigo-600 p-7 rounded-[2rem] shadow-lg shadow-indigo-100 flex flex-col justify-between group hover:bg-indigo-700 transition-colors cursor-pointer" onClick={handleAssessmentGeneration}>
                <p className="text-indigo-100 text-[10px] font-black uppercase tracking-widest">Assessment</p>
                <div className="flex items-center justify-between text-white">
                  <span className="font-bold text-lg leading-tight">Generate<br/>Insight</span>
                  <svg className="w-8 h-8 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 7l5 5m0 0l-5 5m5-5H6"/></svg>
                </div>
              </div>
            </div>

            {/* Trends and Alerts */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="lg:col-span-2 bg-white p-8 rounded-[2.5rem] shadow-sm border border-slate-200 relative overflow-hidden">
                <div className="flex justify-between items-center mb-8">
                  <h3 className="text-xl font-bold text-slate-900 flex items-center gap-3">
                    <div className="w-2 h-6 bg-indigo-600 rounded-full"></div>
                    Longitudinal Trends
                  </h3>
                  <div className="flex gap-2">
                    <span className="px-3 py-1 bg-slate-50 text-slate-500 rounded-lg text-xs font-bold border border-slate-100">Live View</span>
                  </div>
                </div>
                <div className="h-80 -ml-4">
                  {profile.vitals.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={vitalsChartData}>
                        <defs>
                          <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#4f46e5" stopOpacity={0.12}/>
                            <stop offset="95%" stopColor="#4f46e5" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="5 5" vertical={false} stroke="#f1f5f9" />
                        <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 11, fontWeight: 600}} dy={10} />
                        <YAxis hide domain={['auto', 'auto']} />
                        <Tooltip 
                          contentStyle={{borderRadius: '1.5rem', border: 'none', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.1)', padding: '1rem'}}
                          itemStyle={{fontWeight: '800', fontSize: '14px'}}
                        />
                        <Area type="monotone" dataKey="value" stroke="#4f46e5" strokeWidth={4} fillOpacity={1} fill="url(#colorVal)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center text-slate-400">
                      <svg className="w-12 h-12 mb-3 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
                      <p className="font-medium">No vitals data to visualize yet</p>
                    </div>
                  )}
                </div>
              </div>
              
              <div className="bg-white p-8 rounded-[2.5rem] shadow-sm border border-slate-200">
                <h3 className="text-xl font-bold text-slate-900 mb-6">Medication Schedule</h3>
                <div className="space-y-4">
                  {profile.medications.length > 0 ? profile.medications.map((m, i) => (
                    <div key={i} className="flex items-center gap-4 p-5 bg-slate-50/50 rounded-2xl border border-slate-100 hover:border-indigo-100 transition-colors">
                      <div className="w-12 h-12 bg-indigo-50 text-indigo-600 rounded-2xl flex items-center justify-center shrink-0">
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"/></svg>
                      </div>
                      <div className="overflow-hidden">
                        <p className="font-bold text-slate-900 truncate">{m.name}</p>
                        <p className="text-xs text-slate-500 font-medium">{m.dosage} • {m.frequency}</p>
                      </div>
                    </div>
                  )) : (
                    <div className="text-center py-10 opacity-40">
                      <p className="text-slate-400 font-medium italic">Empty pharmacy log</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'assessment' && (
          <div className="space-y-8 animate-fade-in">
            {!profile.assessment ? (
              <div className="bg-white rounded-[3rem] p-16 text-center shadow-sm border border-slate-200">
                <div className="w-24 h-24 bg-indigo-50 rounded-[2rem] flex items-center justify-center mx-auto mb-8 text-indigo-600">
                  <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04M12 21.355r7.106-7.106a11.955 11.955 0 002.512-11.314L12 2.944 1.382 2.935a11.955 11.955 0 002.512 11.314L12 21.355z"/></svg>
                </div>
                <h3 className="text-3xl font-black text-slate-900 mb-4">Clinical Insight Engine</h3>
                <p className="text-slate-500 max-w-lg mx-auto mb-10 text-lg leading-relaxed">Let MedAId analyze your uploaded medical records and vitals to provide a deep clinical assessment of your health risks and progress.</p>
                <button 
                  onClick={handleAssessmentGeneration}
                  disabled={loading}
                  className="bg-indigo-600 hover:bg-indigo-700 text-white px-10 py-5 rounded-3xl font-bold shadow-2xl shadow-indigo-100 transition-all flex items-center gap-3 mx-auto disabled:opacity-50 text-lg"
                >
                  {loading ? 'Synthesizing...' : 'Run Deep Analysis'}
                </button>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2 space-y-8">
                  <div className="bg-white p-10 rounded-[2.5rem] shadow-sm border border-slate-200">
                    <h3 className="text-2xl font-black text-slate-900 mb-6">Patient Executive Summary</h3>
                    <p className="text-slate-600 text-lg leading-relaxed bg-slate-50 p-6 rounded-3xl border border-slate-100">{profile.assessment.summary}</p>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div className="bg-rose-50 p-10 rounded-[2.5rem] border border-rose-100">
                      <h4 className="text-rose-900 font-black text-xl mb-6 flex items-center gap-2">
                        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd"/></svg>
                        Clinical Risk Factors
                      </h4>
                      <ul className="space-y-4">
                        {profile.assessment.risks.map((risk, i) => (
                          <li key={i} className="flex gap-3 text-rose-800 font-medium text-sm">
                            <span className="shrink-0">•</span> {risk}
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div className="bg-indigo-50 p-10 rounded-[2.5rem] border border-indigo-100">
                      <h4 className="text-indigo-900 font-black text-xl mb-6 flex items-center gap-2">
                        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20"><path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z"/><path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd"/></svg>
                        Recommended Actions
                      </h4>
                      <ul className="space-y-4">
                        {profile.assessment.recommendations.map((rec, i) => (
                          <li key={i} className="flex gap-3 text-indigo-800 font-medium text-sm">
                            <span className="shrink-0">•</span> {rec}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
                <div className="bg-slate-900 p-10 rounded-[2.5rem] text-white">
                  <h3 className="text-xl font-black mb-8">Protocol Checklist</h3>
                  <div className="space-y-6">
                    {profile.assessment.nextSteps.map((step, i) => (
                      <div key={i} className="flex gap-4 group">
                        <div className="w-6 h-6 rounded-full border-2 border-slate-700 group-hover:border-indigo-500 transition-colors shrink-0 flex items-center justify-center">
                          <div className="w-2 h-2 rounded-full bg-indigo-500 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                        </div>
                        <p className="text-slate-400 font-medium text-sm leading-relaxed group-hover:text-white transition-colors cursor-default">{step}</p>
                      </div>
                    ))}
                  </div>
                  <div className="mt-12 p-6 bg-slate-800 rounded-3xl border border-slate-700 text-xs font-bold uppercase tracking-widest text-slate-500 text-center">
                    Data last processed: {new Date().toLocaleDateString()}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'history' && (
          <div className="bg-white rounded-[2.5rem] shadow-sm border border-slate-200 overflow-hidden animate-fade-in">
            <div className="p-10 border-b border-slate-100 flex justify-between items-center bg-slate-50/30">
              <h3 className="text-2xl font-black text-slate-900">Historical Clinical Log</h3>
              <div className="flex gap-2">
                <span className="px-4 py-1.5 bg-indigo-50 text-indigo-600 rounded-full text-[10px] font-black uppercase tracking-wider">Secure Database</span>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] border-b border-slate-50">
                    <th className="px-10 py-6">Parameter/Event</th>
                    <th className="px-10 py-6">Reference Value</th>
                    <th className="px-10 py-6">Classification</th>
                    <th className="px-10 py-6 text-right">Observation Date</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-50">
                  {profile.vitals.map(v => (
                    <tr key={v.id} className="hover:bg-slate-50/50 transition-colors group">
                      <td className="px-10 py-6 font-bold text-slate-900 group-hover:text-indigo-600 transition-colors">{v.parameter}</td>
                      <td className="px-10 py-6 text-slate-600 font-medium">{v.reading} <span className="text-slate-400 text-xs">{v.unit}</span></td>
                      <td className="px-10 py-6">
                        <span className={`px-3 py-1.5 rounded-xl text-[10px] font-black uppercase tracking-wider ${v.severity === 'Critical' ? 'bg-rose-100 text-rose-600' : v.severity === 'Elevated' ? 'bg-amber-100 text-amber-600' : 'bg-emerald-100 text-emerald-600'}`}>
                          {v.severity}
                        </span>
                      </td>
                      <td className="px-10 py-6 text-slate-400 text-sm font-medium text-right">{new Date(v.timestamp).toLocaleDateString()}</td>
                    </tr>
                  ))}
                  {profile.history.map(h => (
                    <tr key={h.id} className="hover:bg-slate-50/50 transition-colors group">
                      <td className="px-10 py-6 font-bold text-slate-900 group-hover:text-indigo-600 transition-colors">{h.condition}</td>
                      <td className="px-10 py-6 text-slate-600 font-medium">{h.status}</td>
                      <td className="px-10 py-6">
                        <span className="px-3 py-1.5 bg-slate-100 text-slate-500 rounded-xl text-[10px] font-black uppercase tracking-wider">
                          Historical Record
                        </span>
                      </td>
                      <td className="px-10 py-6 text-slate-400 text-sm font-medium text-right">{h.date}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'chat' && (
          <div className="bg-white rounded-[2.5rem] shadow-sm border border-slate-200 h-[calc(100vh-280px)] flex flex-col animate-fade-in overflow-hidden">
            <div className="p-8 border-b border-slate-100 flex items-center justify-between">
              <h3 className="text-xl font-bold flex items-center gap-3">
                <span className="flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-3 w-3 rounded-full bg-indigo-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-indigo-500"></span>
                </span>
                MedAId Clinical Intelligence
              </h3>
              <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest bg-slate-50 px-3 py-1 rounded-full border border-slate-100">
                End-to-End Encrypted
              </div>
            </div>
            <div className="flex-1 overflow-y-auto p-10 space-y-8 scroll-smooth">
              {chatLog.length === 0 && (
                <div className="text-center py-20">
                  <div className="w-20 h-20 bg-indigo-50 rounded-[2rem] flex items-center justify-center mx-auto mb-6 text-indigo-600">
                    <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"/></svg>
                  </div>
                  <h4 className="text-2xl font-black text-slate-900 mb-2">How can I assist you today?</h4>
                  <p className="text-slate-500 max-w-sm mx-auto font-medium">Ask clinical questions about your results, medications, or general wellness advice based on your profile.</p>
                </div>
              )}
              {chatLog.map((chat, i) => (
                <div key={i} className={`flex ${chat.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] lg:max-w-[70%] px-8 py-5 rounded-[2rem] text-sm md:text-base leading-relaxed ${chat.role === 'user' ? 'bg-slate-900 text-white shadow-2xl shadow-slate-200' : 'bg-slate-100 text-slate-800'}`}>
                    {chat.text}
                  </div>
                </div>
              ))}
              {loading && chatLog.length > 0 && chatLog[chatLog.length - 1].role === 'user' && (
                <div className="flex justify-start">
                   <div className="bg-slate-100 px-8 py-5 rounded-[2rem] flex gap-2">
                     <div className="w-2 h-2 bg-slate-300 rounded-full animate-bounce"></div>
                     <div className="w-2 h-2 bg-slate-300 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                     <div className="w-2 h-2 bg-slate-300 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                   </div>
                </div>
              )}
            </div>
            <form onSubmit={handleChat} className="p-8 bg-white border-t border-slate-100 flex gap-4">
              <input
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Query clinical data..."
                className="flex-1 px-8 py-5 rounded-3xl bg-slate-50 border-none focus:ring-4 focus:ring-indigo-50 focus:bg-white transition-all font-medium text-slate-700 placeholder:text-slate-400"
              />
              <button disabled={loading} className="bg-indigo-600 text-white w-16 h-16 rounded-3xl flex items-center justify-center hover:bg-indigo-700 transition-all shadow-xl shadow-indigo-100 disabled:opacity-50 active:scale-95 shrink-0">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 5l7 7-7 7M5 5l7 7-7 7"/></svg>
              </button>
            </form>
          </div>
        )}

        {activeTab === 'maps' && (
          <div className="space-y-8 animate-fade-in">
            <div className="bg-white p-10 rounded-[2.5rem] shadow-sm border border-slate-200">
              <h3 className="text-2xl font-black text-slate-900 mb-8">Specialized Medical Care Finder</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <button onClick={() => handleMapsSearch('Top Cardiologists')} className="flex flex-col items-center gap-3 p-6 bg-slate-50 hover:bg-indigo-50 hover:text-indigo-600 rounded-[2rem] border border-slate-100 transition-all font-bold text-sm text-slate-600 group active:scale-95">
                  <div className="w-10 h-10 bg-white rounded-xl shadow-sm flex items-center justify-center group-hover:scale-110 transition-transform">❤️</div>
                  Cardiology
                </button>
                <button onClick={() => handleMapsSearch('Urgent Care Hospitals')} className="flex flex-col items-center gap-3 p-6 bg-slate-50 hover:bg-rose-50 hover:text-rose-600 rounded-[2rem] border border-slate-100 transition-all font-bold text-sm text-slate-600 group active:scale-95">
                  <div className="w-10 h-10 bg-white rounded-xl shadow-sm flex items-center justify-center group-hover:scale-110 transition-transform">🏥</div>
                  Emergency
                </button>
                <button onClick={() => handleMapsSearch('Phlebotomy Labs')} className="flex flex-col items-center gap-3 p-6 bg-slate-50 hover:bg-emerald-50 hover:text-emerald-600 rounded-[2rem] border border-slate-100 transition-all font-bold text-sm text-slate-600 group active:scale-95">
                   <div className="w-10 h-10 bg-white rounded-xl shadow-sm flex items-center justify-center group-hover:scale-110 transition-transform">🧪</div>
                  Labs
                </button>
                <button onClick={() => handleMapsSearch('Pharmacies')} className="flex flex-col items-center gap-3 p-6 bg-slate-50 hover:bg-amber-50 hover:text-amber-600 rounded-[2rem] border border-slate-100 transition-all font-bold text-sm text-slate-600 group active:scale-95">
                   <div className="w-10 h-10 bg-white rounded-xl shadow-sm flex items-center justify-center group-hover:scale-110 transition-transform">💊</div>
                  Pharmacy
                </button>
              </div>
            </div>
            
            {mapsResults && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-white p-10 rounded-[2.5rem] shadow-sm border border-slate-200">
                  <h4 className="font-black text-slate-900 text-xl mb-6">Service Overview</h4>
                  <div className="text-slate-600 leading-relaxed font-medium whitespace-pre-wrap prose prose-indigo prose-sm max-w-none">
                    {mapsResults.text}
                  </div>
                </div>
                <div className="space-y-4">
                  <h4 className="font-black text-slate-900 text-xl mb-6 px-2">Verified Facilities</h4>
                  {mapsResults.links.map((link, i) => (
                    <a key={i} href={link.uri} target="_blank" rel="noreferrer" className="flex items-center justify-between p-7 bg-white border border-slate-200 rounded-[2rem] hover:border-indigo-500 hover:shadow-2xl transition-all group hover:-translate-y-1">
                      <div>
                        <p className="font-black text-slate-900 text-lg group-hover:text-indigo-600 transition-colors">{link.title}</p>
                        <p className="text-[10px] text-indigo-600 font-black uppercase tracking-widest mt-2 flex items-center gap-2">
                          Google Maps Navigation
                          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
                        </p>
                      </div>
                      <div className="w-12 h-12 bg-indigo-50 text-indigo-600 rounded-2xl flex items-center justify-center group-hover:bg-indigo-600 group-hover:text-white transition-all">
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"/><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
                      </div>
                    </a>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
};

// --- RENDER ---
const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<App />);
