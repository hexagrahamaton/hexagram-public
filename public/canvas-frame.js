class CanvasFrame {
  constructor(name, args={}) {
    let defaults = {
      canvas: args.canvas || document.createElement('canvas'),
      img: args.img || new Image(),
      dim: 1440,
      session: args.session || Session.get(),
      drawFn: null,
    };
    Object.assign(this, defaults, args);
    this.name = name;
    if (typeof this.dim == 'number')
      this.dim = [this.dim, this.dim];

    this.canvas.width = this.dim[0];
    this.canvas.height = this.dim[1];
    this.ctx = this.canvas.getContext('2d');
    this.ctx.translate(this.dim[0] / 2, this.dim[1] / 2);
    this.loadImageFromDb();

    this.img.onload = () => this.handleOnload();
  }

  async loadImageFromDb() {
    let obj = await this.session.db.media.get(this.name);
    if (obj) {
      let data = new Uint8Array(obj.data);
      let blob = new Blob([data], {type: obj.fileType});
      let earl = window.URL.createObjectURL(blob);
      this.img.src = earl;
    }
  }

  loadImageFromPrompt(ctx) {
    const reader = new FileReader();
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.multiple = false;
    let file, fileType, fileName;

    input.onchange = () => {
      file = input.files[0];
      fileType = file.type;
      fileName = file.name;
      reader.readAsArrayBuffer(file);
    };

    reader.onload = (ev) => {
      let data = ev.target.result;
      data = new Uint8Array(data);
      let blob = new Blob([data], {type: fileType});
      let earl = window.URL.createObjectURL(blob);
      this.img.src = earl;
      let obj = {
        name: this.name,
        data: data,
        type: fileType
      }
      let x = this.session.db.media.put(obj);
    };

    input.click();
  }

  handleOnload() {
    let img = this.img;
    let [w, h] = [img.width, img.height];
    let r = w/h;
    let d = this.dim[0];
    if (w < h) {
      w = d;
      h = d/r;
    }
    else {
      h = d;
      w = d*r;
    }
    this.ctx.drawImage(img, -w/2, -h/2, w, h);
  }

  draw() {
    this.drawFn && this.drawFn();
  }

}
