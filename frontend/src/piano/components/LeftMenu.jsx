export default function LeftMenu({ isOpen, onToggle, onLibraryOpen, libraryCount = 0 }) {
  return (
    <div className={`left-menu ${isOpen ? 'open' : ''}`}>
      <div className="left-menu-content">
        <div className="left-menu-header">
          <span className="left-menu-logo">
            <img className="left-menu-brand-icon" src="/piano-iq-icon.svg" alt="" aria-hidden="true" />
            <span>
              <strong>Piano IQ</strong>
              <small>Intelligent Piano Learning System</small>
            </span>
          </span>

          <button
            className="panel-close-btn"
            onClick={onToggle}
            title="Close menu"
            type="button"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" />
            </svg>
          </button>
        </div>

        <nav className="left-menu-nav">
          <a href="#/" className="left-menu-link">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z" />
            </svg>
            <span>Home</span>
          </a>
          <a href="#" className="left-menu-link disabled">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
            </svg>
            <span>Profile</span>
            <span className="coming-soon">Soon</span>
          </a>
          <button type="button" className="left-menu-link" onClick={onLibraryOpen}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M20 2H8c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 5h-3v5.5a2.5 2.5 0 0 1-5 0 2.5 2.5 0 0 1 2.5-2.5c.57 0 1.08.19 1.5.51V5h4v2zM4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6z" />
            </svg>
            <span>Library</span>
            {libraryCount > 0 && <span className="coming-soon">{libraryCount}</span>}
          </button>
        </nav>

        <div className="left-menu-footer">
          <p className="left-menu-hint">Piano IQ Tutor v0.1</p>
        </div>
      </div>
    </div>
  );
}
