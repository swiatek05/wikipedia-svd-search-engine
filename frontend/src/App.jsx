import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import './App.css';

function SearchPage({ query, setQuery, results, setResults, isLoading, setIsLoading }) {
    const navigate = useNavigate();

    const handleSearch = async () => {
        if (!query.trim()) return;
        setIsLoading(true);
        try {
            const response = await fetch(`http://localhost:5000/api/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query }),
            });
            const data = await response.json();
            setResults(data);
        } catch {
            alert("Błąd połączenia z serwerem Pythona. Upewnij się, że Flask działa!");
        } finally {
            setIsLoading(false);
        }
    };

    const readArticle = async (title) => {
        try {
            const response = await fetch(`http://localhost:5000/api/article`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title }),
            });
            const data = await response.json();

            navigate('/article', { state: { title: title, content: data.content } });
        } catch {
            alert("Błąd pobrania treści artykułu");
        }
    };

    return (
        <div className="container">
            <header>
                <h1>Wyszukiwarka SVD vs TF-IDF</h1>
                <div className="search-box">
                    <input
                        type="text"
                        value={query}
                        onChange={e => setQuery(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                        placeholder="Czego szukasz?"
                    />
                    <button onClick={handleSearch} disabled={isLoading}>
                        {isLoading ? 'Przetwarzanie SVD...' : 'Szukaj'}
                    </button>
                </div>
            </header>

            <main className="result-grid">

                <section className="column">
                    <h2>Klasyczne (TF-IDF)</h2>
                    <div className="results-list">
                        {results.tfidf.map((item, i) => (
                            <div key={i} className="result-card" onClick={() => readArticle(item.title)}>
                                <h3>{item.title}</h3>
                                <span className="score">Podobieństwo: {item.score.toFixed(4)}</span>
                            </div>
                        ))}
                    </div>
                </section>


                <section className="column">
                    <h2 className="svd-title">Semantyczne (SVD)</h2>
                    <div className="results-list">
                        {results.svd_LRA.map((item, i) => (
                            <div key={i} className="result-card svd-card" onClick={() => readArticle(item.title)}>
                                <h3>{item.title}</h3>
                                <span className="score">Podobieństwo: {item.score.toFixed(4)}</span>
                            </div>
                        ))}
                    </div>
                </section>
            </main>
        </div>
    );
}


function ArticlePage() {
    const navigate = useNavigate();
    const location = useLocation();
    const article = location.state;

    if (!article) return <div className="container"><p>Nie wybrano artykułu.</p></div>;

    return (
        <div className="container article-page">
            <button className="back-btn" onClick={() => navigate(-1)}>⬅ Wróć do wyników</button>
            <article>
                <h1>{article.title}</h1>
                <div className="content-box">
                    {article.content}
                </div>
            </article>
        </div>
    );
}

function App() {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState({ tfidf: [], svd_LRA: [] });
    const [isLoading, setIsLoading] = useState(false);

    return (
        <Router>
            <Routes>
                <Route path="/" element={
                    <SearchPage
                        query={query} setQuery={setQuery}
                        results={results} setResults={setResults}
                        isLoading={isLoading} setIsLoading={setIsLoading}
                    />
                } />
                <Route path="/article" element={<ArticlePage />} />
            </Routes>
        </Router>
    );
}

export default App;