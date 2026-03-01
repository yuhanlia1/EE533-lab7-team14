`timescale 1ns/1ps

`define UDP_REG_ADDR_WIDTH 23
`define CPCI_NF2_DATA_WIDTH 32

module gpu_datapath #(
  parameter UDP_REG_SRC_WIDTH = 2,
  parameter IMEM_DEPTH = 256,
  parameter DMEM_DEPTH = 1024,
  parameter IMEM_AW = 8,
  parameter DMEM_AW = 10,
  parameter GPU_ADDR_PREFIX = 8'h7F
)(
  input                               clk,
  input                               reset,

  input                               reg_req_in,
  input                               reg_ack_in,
  input                               reg_rd_wr_L_in,
  input  [`UDP_REG_ADDR_WIDTH-1:0]    reg_addr_in,
  input  [`CPCI_NF2_DATA_WIDTH-1:0]   reg_data_in,
  input  [UDP_REG_SRC_WIDTH-1:0]      reg_src_in,

  output reg                          reg_req_out,
  output reg                          reg_ack_out,
  output reg                          reg_rd_wr_L_out,
  output reg [`UDP_REG_ADDR_WIDTH-1:0]   reg_addr_out,
  output reg [`CPCI_NF2_DATA_WIDTH-1:0]  reg_data_out,
  output reg [UDP_REG_SRC_WIDTH-1:0]     reg_src_out
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

  wire [7:0] addr_prefix;
  wire [7:0] addr_off;
  wire addr_hit;

  assign addr_prefix = reg_addr_in[22:15];
  assign addr_off    = reg_addr_in[7:0];
  assign addr_hit = (addr_prefix == GPU_ADDR_PREFIX);

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

  reg run_reg;

  wire [IMEM_AW-1:0] dbg_pc;
  wire [IMEM_AW-1:0] dbg_pcf;
  wire [31:0]        dbg_ir;
  wire [9:0]         dbg_tid;
  wire               dbg_ld_wait;
  wire               dbg_pending;
  wire [DMEM_AW-1:0] dbg_eff_addr;
  wire               dbg_rf_we;
  wire [4:0]         dbg_rf_rd;
  wire [63:0]        dbg_rf_wdata;
  wire               dbg_st_we;
  wire [DMEM_AW-1:0] dbg_st_addr;
  wire [63:0]        dbg_st_data;

  wire [IMEM_AW-1:0] core_imem_raddr;
  wire [31:0]        core_imem_rdata;

  wire core_dmem_we;
  wire [DMEM_AW-1:0] core_dmem_waddr;
  wire [63:0]        core_dmem_wdata;
  wire [DMEM_AW-1:0] core_dmem_raddr;
  wire [63:0]        core_dmem_rdata;
  wire [63:0]        dmem_dbg_rdata;

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
    .rdata(core_dmem_rdata),
    .dbg_raddr(dmem_addr),
    .dbg_rdata(dmem_dbg_rdata)
  );

  gpu_core #(
    .IMEM_DEPTH(IMEM_DEPTH),
    .DMEM_DEPTH(DMEM_DEPTH),
    .IMEM_AW(IMEM_AW),
    .DMEM_AW(DMEM_AW)
  ) u_core (
    .clk(clk),
    .rst(reset),
    .run(run_reg),
    .n_words(n_words[9:0]),
    .a_base(a_base[DMEM_AW-1:0]),
    .b_base(b_base[DMEM_AW-1:0]),
    .c_base(c_base[DMEM_AW-1:0]),
    .done(),

    .dbg_pc(dbg_pc),
    .dbg_pcf(dbg_pcf),
    .dbg_ir(dbg_ir),
    .dbg_tid(dbg_tid),
    .dbg_ld_wait(dbg_ld_wait),
    .dbg_pending(dbg_pending),
    .dbg_eff_addr(dbg_eff_addr),
    .dbg_rf_we(dbg_rf_we),
    .dbg_rf_rd(dbg_rf_rd),
    .dbg_rf_wdata(dbg_rf_wdata),
    .dbg_st_we(dbg_st_we),
    .dbg_st_addr(dbg_st_addr),
    .dbg_st_data(dbg_st_data),

    .imem_rdata(core_imem_rdata),
    .imem_raddr(core_imem_raddr),
    .dmem_we(core_dmem_we),
    .dmem_waddr(core_dmem_waddr),
    .dmem_wdata(core_dmem_wdata),
    .dmem_raddr(core_dmem_raddr),
    .dmem_rdata(core_dmem_rdata)
  );

  wire handle_req;
  assign handle_req = reg_req_in & (~reg_ack_in) & addr_hit;

  reg [31:0] rd_data;

  always @(*) begin
    rd_data = 32'h0;
    case (addr_off)
      OFF_CONTROL:    rd_data = {31'd0, run_reg};
      OFF_A_BASE:     rd_data = a_base;
      OFF_B_BASE:     rd_data = b_base;
      OFF_C_BASE:     rd_data = c_base;
      OFF_N_WORDS:    rd_data = n_words;
      OFF_IMEM_ADDR:  rd_data = {{(32-IMEM_AW){1'b0}}, imem_addr};
      OFF_IMEM_WDATA: rd_data = imem_wdata;
      OFF_DMEM_ADDR:  rd_data = {{(32-DMEM_AW){1'b0}}, dmem_addr};
      OFF_DMEM_WLO:   rd_data = dmem_wlo;
      OFF_DMEM_WHI:   rd_data = dmem_whi;
      OFF_DMEM_RLO:   rd_data = dmem_dbg_rdata[31:0];
      OFF_DMEM_RHI:   rd_data = dmem_dbg_rdata[63:32];
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
      run_reg <= 1'b0;
    end else begin
      imem_we_pulse <= 1'b0;
      dmem_we_pulse <= 1'b0;

      if (handle_req && (reg_rd_wr_L_in == 1'b0)) begin
        case (addr_off)
          OFF_CONTROL:    run_reg <= reg_data_in[0];
          OFF_A_BASE:     a_base <= reg_data_in;
          OFF_B_BASE:     b_base <= reg_data_in;
          OFF_C_BASE:     c_base <= reg_data_in;
          OFF_N_WORDS:    n_words <= reg_data_in;
          OFF_IMEM_ADDR:  imem_addr <= reg_data_in[IMEM_AW-1:0];
          OFF_IMEM_WDATA: imem_wdata <= reg_data_in;
          OFF_IMEM_WE:    imem_we_pulse <= reg_data_in[0];
          OFF_DMEM_ADDR:  dmem_addr <= reg_data_in[DMEM_AW-1:0];
          OFF_DMEM_WLO:   dmem_wlo <= reg_data_in;
          OFF_DMEM_WHI:   dmem_whi <= reg_data_in;
          OFF_DMEM_WE:    dmem_we_pulse <= reg_data_in[0];
          default: begin end
        endcase
      end
    end
  end

endmodule
