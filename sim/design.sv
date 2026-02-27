`timescale 1ns/1ps

`define UDP_REG_ADDR_WIDTH 23
`define CPCI_NF2_DATA_WIDTH 32

module gpu_pc #(parameter PC_W = 8) (
  input               clk,
  input               rst,
  input               en,
  input  [PC_W-1:0]    pc_next,
  output reg [PC_W-1:0] pc
);
  always @(posedge clk) begin
    if (rst) pc <= {PC_W{1'b0}};
    else if (en) pc <= pc_next;
  end
endmodule

module gpu_tid #(parameter TID_W = 10) (
  input               clk,
  input               rst,
  input               en_set,
  input               en_inc,
  input  [TID_W-1:0]   tid_set,
  output reg [TID_W-1:0] tid
);
  always @(posedge clk) begin
    if (rst) tid <= {TID_W{1'b0}};
    else begin
      if (en_set) tid <= tid_set;
      else if (en_inc) tid <= tid + {{(TID_W-1){1'b0}},1'b1};
    end
  end
endmodule

module gpu_regfile64 #(parameter NREG = 8) (
  input               clk,
  input               we,
  input  [2:0]        waddr,
  input  [63:0]       wdata,
  input  [2:0]        raddr1,
  input  [2:0]        raddr2,
  output [63:0]       rdata1,
  output [63:0]       rdata2
);
  reg [63:0] regs [0:NREG-1];
  assign rdata1 = regs[raddr1];
  assign rdata2 = regs[raddr2];
  always @(posedge clk) begin
    if (we) regs[waddr] <= wdata;
  end
endmodule

module gpu_exec_simd (
  input  [1:0]  op,
  input  [63:0] a,
  input  [63:0] b,
  output [63:0] y
);
  wire signed [15:0] a0 = a[15:0];
  wire signed [15:0] a1 = a[31:16];
  wire signed [15:0] a2 = a[47:32];
  wire signed [15:0] a3 = a[63:48];

  wire signed [15:0] b0 = b[15:0];
  wire signed [15:0] b1 = b[31:16];
  wire signed [15:0] b2 = b[47:32];
  wire signed [15:0] b3 = b[63:48];

  reg [15:0] y0, y1, y2, y3;

  always @(*) begin
    case (op)
      2'd0: begin y0 = a0 + b0; y1 = a1 + b1; y2 = a2 + b2; y3 = a3 + b3; end
      2'd1: begin y0 = a0 - b0; y1 = a1 - b1; y2 = a2 - b2; y3 = a3 - b3; end
      2'd2: begin y0 = a0 * b0; y1 = a1 * b1; y2 = a2 * b2; y3 = a3 * b3; end
      default: begin y0 = 16'h0; y1 = 16'h0; y2 = 16'h0; y3 = 16'h0; end
    endcase
  end

  assign y = {y3, y2, y1, y0};
endmodule

module gpu_tensor_dot4_i16 (
  input  [63:0] a,
  input  [63:0] b,
  output [63:0] y
);
  wire signed [15:0] a0 = a[15:0];
  wire signed [15:0] a1 = a[31:16];
  wire signed [15:0] a2 = a[47:32];
  wire signed [15:0] a3 = a[63:48];

  wire signed [15:0] b0 = b[15:0];
  wire signed [15:0] b1 = b[31:16];
  wire signed [15:0] b2 = b[47:32];
  wire signed [15:0] b3 = b[63:48];

  wire signed [31:0] p0 = a0 * b0;
  wire signed [31:0] p1 = a1 * b1;
  wire signed [31:0] p2 = a2 * b2;
  wire signed [31:0] p3 = a3 * b3;

  wire signed [31:0] sum = p0 + p1 + p2 + p3;
  assign y = {32'h0, sum[31:0]};
endmodule

module gpu_imem #(parameter DEPTH = 256, parameter AW = 8) (
  input               clk,
  input  [AW-1:0]      raddr,
  output [31:0]        rdata,
  input               we,
  input  [AW-1:0]      waddr,
  input  [31:0]        wdata
);
  reg [31:0] mem [0:DEPTH-1];
  assign rdata = mem[raddr];
  always @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
  end
endmodule

module gpu_dmem #(parameter DEPTH = 1024, parameter AW = 10) (
  input               clk,
  input               we,
  input  [AW-1:0]      waddr,
  input  [63:0]        wdata,
  input  [AW-1:0]      raddr,
  output [63:0]        rdata
);
  reg [63:0] mem [0:DEPTH-1];
  assign rdata = mem[raddr];
  always @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
  end
endmodule

module gpu_core #(
  parameter IMEM_DEPTH = 256,
  parameter DMEM_DEPTH = 1024,
  parameter IMEM_AW = 8,
  parameter DMEM_AW = 10,
  parameter TID_W = 10
)(
  input               clk,
  input               rst,
  input               start,
  input  [31:0]        a_base,
  input  [31:0]        b_base,
  input  [31:0]        c_base,
  input  [31:0]        n_words,
  output reg          busy,
  output reg          done,
  output [IMEM_AW-1:0] dbg_pc,
  output [31:0]        dbg_ir,
  output [TID_W-1:0]   dbg_tid,
  output reg           dbg_st_we,
  output reg [DMEM_AW-1:0] dbg_st_addr,
  output reg [63:0]    dbg_st_data,
  output              imem_we,
  output [IMEM_AW-1:0] imem_waddr,
  output [31:0]        imem_wdata,
  output [IMEM_AW-1:0] imem_raddr,
  input  [31:0]        imem_rdata,
  output              dmem_we,
  output [DMEM_AW-1:0] dmem_waddr,
  output [63:0]        dmem_wdata,
  output [DMEM_AW-1:0] dmem_raddr,
  input  [63:0]        dmem_rdata
);

  wire core_rst = rst | start;

  wire [IMEM_AW-1:0] pc;
  reg  [IMEM_AW-1:0] pc_next;
  reg                pc_en;

  wire [TID_W-1:0] tid;
  reg              tid_set_en;
  reg              tid_inc_en;
  reg  [TID_W-1:0] tid_set_val;

  gpu_pc #(.PC_W(IMEM_AW)) u_pc (
    .clk(clk), .rst(core_rst),
    .en(pc_en),
    .pc_next(pc_next),
    .pc(pc)
  );

  gpu_tid #(.TID_W(TID_W)) u_tid (
    .clk(clk), .rst(core_rst),
    .en_set(tid_set_en),
    .en_inc(tid_inc_en),
    .tid_set(tid_set_val),
    .tid(tid)
  );

  assign imem_raddr = pc;
  wire [31:0] ir = imem_rdata;

  wire [3:0]  op   = ir[31:28];
  wire [2:0]  rd   = ir[27:25];
  wire [2:0]  rs1  = ir[24:22];
  wire [2:0]  rs2  = ir[21:19];
  wire [1:0]  bsel = ir[18:17];
  wire signed [15:0] imm16 = ir[15:0];

  wire [63:0] rf_r1, rf_r2;
  reg         rf_we;
  reg  [2:0]  rf_waddr;
  reg  [63:0] rf_wdata;

  gpu_regfile64 u_rf (
    .clk(clk),
    .we(rf_we),
    .waddr(rf_waddr),
    .wdata(rf_wdata),
    .raddr1(rs1),
    .raddr2(rs2),
    .rdata1(rf_r1),
    .rdata2(rf_r2)
  );

  wire [63:0] simd_add;
  gpu_exec_simd u_simd (
    .op(2'd0),
    .a(rf_r1),
    .b(rf_r2),
    .y(simd_add)
  );

  wire [63:0] dot4;
  gpu_tensor_dot4_i16 u_dot (
    .a(rf_r1),
    .b(rf_r2),
    .y(dot4)
  );

  wire [31:0] tid_ext = {{(32-TID_W){1'b0}}, tid};
  wire signed [31:0] imm_ext = {{16{imm16[15]}}, imm16};

  reg [31:0] base_sel_val;
  always @(*) begin
    case (bsel)
      2'd0: base_sel_val = a_base;
      2'd1: base_sel_val = b_base;
      2'd2: base_sel_val = c_base;
      default: base_sel_val = 32'd0;
    endcase
  end

  wire [31:0] eff_addr_word = base_sel_val + tid_ext + imm_ext;

  assign dbg_pc = pc;
  assign dbg_ir = ir;
  assign dbg_tid = tid;

  assign imem_we = 1'b0;
  assign imem_waddr = {IMEM_AW{1'b0}};
  assign imem_wdata = 32'h0;

  reg dmem_we_r;
  reg [DMEM_AW-1:0] dmem_waddr_r;
  reg [63:0] dmem_wdata_r;
  reg [DMEM_AW-1:0] dmem_raddr_r;

  assign dmem_we = dmem_we_r;
  assign dmem_waddr = dmem_waddr_r;
  assign dmem_wdata = dmem_wdata_r;
  assign dmem_raddr = dmem_raddr_r;

  always @(*) begin
    pc_next = pc + {{(IMEM_AW-1){1'b0}},1'b1};
    pc_en = 1'b0;

    tid_set_en = 1'b0;
    tid_inc_en = 1'b0;
    tid_set_val = {TID_W{1'b0}};

    rf_we = 1'b0;
    rf_waddr = rd;
    rf_wdata = 64'h0;

    dmem_we_r = 1'b0;
    dmem_waddr_r = eff_addr_word[DMEM_AW-1:0];
    dmem_wdata_r = 64'h0;
    dmem_raddr_r = eff_addr_word[DMEM_AW-1:0];

    dbg_st_we = 1'b0;
    dbg_st_addr = eff_addr_word[DMEM_AW-1:0];
    dbg_st_data = 64'h0;

    if (busy) begin
      pc_en = 1'b1;
      case (op)
        4'h0: begin end
        4'h1: begin
          rf_we = 1'b1;
          rf_wdata = {{48{1'b0}}, ir[15:0]};
        end
        4'h2: begin
          rf_we = 1'b1;
          rf_wdata = dmem_rdata;
        end
        4'h3: begin
          dmem_we_r = 1'b1;
          dmem_wdata_r = rf_r2;
          dbg_st_we = 1'b1;
          dbg_st_data = rf_r2;
        end
        4'h4: begin
          rf_we = 1'b1;
          rf_wdata = simd_add;
        end
        4'h6: begin
          rf_we = 1'b1;
          rf_wdata = dot4;
        end
        4'h8: begin
          tid_set_en = 1'b1;
          tid_set_val = ir[9:0];
        end
        4'h9: begin
          tid_inc_en = 1'b1;
        end
        4'hA: begin
          if (tid_ext < n_words) pc_next = ir[7:0];
        end
        4'hF: begin end
        default: begin end
      endcase
    end
  end

  always @(posedge clk) begin
    if (rst) begin
      busy <= 1'b0;
      done <= 1'b0;
    end else begin
      if (start) begin
        busy <= 1'b1;
        done <= 1'b0;
      end else if (busy && op == 4'hF) begin
        busy <= 1'b0;
        done <= 1'b1;
      end
    end
  end

endmodule

module gpu_datapath #(
  parameter UDP_REG_SRC_WIDTH = 2,
  parameter IMEM_DEPTH = 256,
  parameter DMEM_DEPTH = 1024,
  parameter IMEM_AW = 8,
  parameter DMEM_AW = 10,
  parameter GPU_ADDR_PREFIX = 8'h7F
)(
  input                              clk,
  input                              reset,

  input                              reg_req_in,
  input                              reg_ack_in,
  input                              reg_rd_wr_L_in,
  input  [`UDP_REG_ADDR_WIDTH-1:0]   reg_addr_in,
  input  [`CPCI_NF2_DATA_WIDTH-1:0]  reg_data_in,
  input  [UDP_REG_SRC_WIDTH-1:0]     reg_src_in,

  output reg                         reg_req_out,
  output reg                         reg_ack_out,
  output reg                         reg_rd_wr_L_out,
  output reg [`UDP_REG_ADDR_WIDTH-1:0]  reg_addr_out,
  output reg [`CPCI_NF2_DATA_WIDTH-1:0] reg_data_out,
  output reg [UDP_REG_SRC_WIDTH-1:0]    reg_src_out
);

  localparam OFF_CONTROL     = 8'h00;
  localparam OFF_A_BASE      = 8'h01;
  localparam OFF_B_BASE      = 8'h02;
  localparam OFF_C_BASE      = 8'h03;
  localparam OFF_N_WORDS     = 8'h04;
  localparam OFF_IMEM_ADDR   = 8'h10;
  localparam OFF_IMEM_WDATA  = 8'h11;
  localparam OFF_IMEM_WE     = 8'h12;
  localparam OFF_DMEM_ADDR   = 8'h20;
  localparam OFF_DMEM_WLO    = 8'h21;
  localparam OFF_DMEM_WHI    = 8'h22;
  localparam OFF_DMEM_WE     = 8'h23;
  localparam OFF_DMEM_RLO    = 8'h24;
  localparam OFF_DMEM_RHI    = 8'h25;
  localparam OFF_DBG_PC      = 8'h30;
  localparam OFF_DBG_IR      = 8'h31;
  localparam OFF_DBG_TID     = 8'h32;
  localparam OFF_DBG_LASTST  = 8'h33;

  wire [7:0] addr_prefix = reg_addr_in[22:15];
  wire [7:0] addr_off    = reg_addr_in[7:0];
  wire addr_hit = (addr_prefix == GPU_ADDR_PREFIX);

  reg [31:0] a_base;
  reg [31:0] b_base;
  reg [31:0] c_base;
  reg [31:0] n_words;

  reg [IMEM_AW-1:0] imem_addr;
  reg [31:0]        imem_wdata;
  reg               imem_we_pulse;

  reg [DMEM_AW-1:0] dmem_addr;
  reg [31:0]        dmem_wlo;
  reg [31:0]        dmem_whi;
  reg               dmem_we_pulse;

  reg start_pulse;

  wire busy;
  wire done;

  wire [IMEM_AW-1:0] dbg_pc;
  wire [31:0]        dbg_ir;
  wire [9:0]         dbg_tid;
  wire               dbg_st_we;
  wire [DMEM_AW-1:0] dbg_st_addr;
  wire [63:0]        dbg_st_data;

  wire core_imem_we;
  wire [IMEM_AW-1:0] core_imem_waddr;
  wire [31:0] core_imem_wdata;
  wire [IMEM_AW-1:0] core_imem_raddr;
  wire [31:0] core_imem_rdata;

  wire core_dmem_we;
  wire [DMEM_AW-1:0] core_dmem_waddr;
  wire [63:0] core_dmem_wdata;
  wire [DMEM_AW-1:0] core_dmem_raddr;
  wire [63:0] core_dmem_rdata;

  gpu_imem #(.DEPTH(IMEM_DEPTH), .AW(IMEM_AW)) u_imem (
    .clk(clk),
    .raddr(core_imem_raddr),
    .rdata(core_imem_rdata),
    .we(imem_we_pulse),
    .waddr(imem_addr),
    .wdata(imem_wdata)
  );

  gpu_dmem #(.DEPTH(DMEM_DEPTH), .AW(DMEM_AW)) u_dmem (
    .clk(clk),
    .we(dmem_we_pulse | core_dmem_we),
    .waddr(dmem_we_pulse ? dmem_addr : core_dmem_waddr),
    .wdata(dmem_we_pulse ? {dmem_whi, dmem_wlo} : core_dmem_wdata),
    .raddr(core_dmem_raddr),
    .rdata(core_dmem_rdata)
  );

  gpu_core #(
    .IMEM_DEPTH(IMEM_DEPTH),
    .DMEM_DEPTH(DMEM_DEPTH),
    .IMEM_AW(IMEM_AW),
    .DMEM_AW(DMEM_AW),
    .TID_W(10)
  ) u_core (
    .clk(clk),
    .rst(reset),
    .start(start_pulse),
    .a_base(a_base),
    .b_base(b_base),
    .c_base(c_base),
    .n_words(n_words),
    .busy(busy),
    .done(done),
    .dbg_pc(dbg_pc),
    .dbg_ir(dbg_ir),
    .dbg_tid(dbg_tid),
    .dbg_st_we(dbg_st_we),
    .dbg_st_addr(dbg_st_addr),
    .dbg_st_data(dbg_st_data),
    .imem_we(core_imem_we),
    .imem_waddr(core_imem_waddr),
    .imem_wdata(core_imem_wdata),
    .imem_raddr(core_imem_raddr),
    .imem_rdata(core_imem_rdata),
    .dmem_we(core_dmem_we),
    .dmem_waddr(core_dmem_waddr),
    .dmem_wdata(core_dmem_wdata),
    .dmem_raddr(core_dmem_raddr),
    .dmem_rdata(core_dmem_rdata)
  );

  wire handle_req = reg_req_in & (~reg_ack_in) & addr_hit;

  reg [31:0] rd_data;

  always @(*) begin
    rd_data = 32'h0;
    case (addr_off)
      OFF_CONTROL:    rd_data = {30'd0, done, busy};
      OFF_A_BASE:     rd_data = a_base;
      OFF_B_BASE:     rd_data = b_base;
      OFF_C_BASE:     rd_data = c_base;
      OFF_N_WORDS:    rd_data = n_words;
      OFF_IMEM_ADDR:  rd_data = {{(32-IMEM_AW){1'b0}}, imem_addr};
      OFF_IMEM_WDATA: rd_data = imem_wdata;
      OFF_DMEM_ADDR:  rd_data = {{(32-DMEM_AW){1'b0}}, dmem_addr};
      OFF_DMEM_WLO:   rd_data = dmem_wlo;
      OFF_DMEM_WHI:   rd_data = dmem_whi;
      OFF_DMEM_RLO:   rd_data = u_dmem.mem[dmem_addr][31:0];
      OFF_DMEM_RHI:   rd_data = u_dmem.mem[dmem_addr][63:32];
      OFF_DBG_PC:     rd_data = {{24{1'b0}}, dbg_pc};
      OFF_DBG_IR:     rd_data = dbg_ir;
      OFF_DBG_TID:    rd_data = {{22{1'b0}}, dbg_tid};
      OFF_DBG_LASTST: rd_data = {{(32-DMEM_AW){1'b0}}, dbg_st_addr};
      default:        rd_data = 32'h0;
    endcase
  end

  always @(*) begin
    reg_req_out = reg_req_in;
    reg_ack_out = reg_ack_in;
    reg_rd_wr_L_out = reg_rd_wr_L_in;
    reg_addr_out = reg_addr_in;
    reg_data_out = reg_data_in;
    reg_src_out  = reg_src_in;

    if (handle_req) begin
      reg_ack_out = 1'b1;
      if (reg_rd_wr_L_in) reg_data_out = rd_data;
      else reg_data_out = 32'h0;
    end
  end

  always @(posedge clk) begin
    if (reset) begin
      a_base <= 32'd0;
      b_base <= 32'd0;
      c_base <= 32'd0;
      n_words <= 32'd0;
      imem_addr <= {IMEM_AW{1'b0}};
      imem_wdata <= 32'd0;
      imem_we_pulse <= 1'b0;
      dmem_addr <= {DMEM_AW{1'b0}};
      dmem_wlo <= 32'd0;
      dmem_whi <= 32'd0;
      dmem_we_pulse <= 1'b0;
      start_pulse <= 1'b0;
    end else begin
      imem_we_pulse <= 1'b0;
      dmem_we_pulse <= 1'b0;
      start_pulse <= 1'b0;

      if (handle_req && (reg_rd_wr_L_in == 1'b0)) begin
        case (addr_off)
          OFF_CONTROL:   start_pulse <= reg_data_in[0];
          OFF_A_BASE:    a_base <= reg_data_in;
          OFF_B_BASE:    b_base <= reg_data_in;
          OFF_C_BASE:    c_base <= reg_data_in;
          OFF_N_WORDS:   n_words <= reg_data_in;
          OFF_IMEM_ADDR: imem_addr <= reg_data_in[IMEM_AW-1:0];
          OFF_IMEM_WDATA: imem_wdata <= reg_data_in;
          OFF_IMEM_WE:   imem_we_pulse <= reg_data_in[0];
          OFF_DMEM_ADDR: dmem_addr <= reg_data_in[DMEM_AW-1:0];
          OFF_DMEM_WLO:  dmem_wlo <= reg_data_in;
          OFF_DMEM_WHI:  dmem_whi <= reg_data_in;
          OFF_DMEM_WE:   dmem_we_pulse <= reg_data_in[0];
          default: begin end
        endcase
      end
    end
  end

endmodule