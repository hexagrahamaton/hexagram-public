class Session {
  static get() {
    Session.instance = Session.instance || new Session();
    return Session.instance;
  }

  constructor() {
    const DB_VERSION = 1;
    const DB_SCHEME = {
      media: {keyPath: 'name'}
    };
    this.db = new Database('hexagram', DB_VERSION, DB_SCHEME);
    this.db.connect();
  }
}
